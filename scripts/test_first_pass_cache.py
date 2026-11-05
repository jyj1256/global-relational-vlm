import sys
from pathlib import Path

import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from vlm.data.gqa_dataset import GQADataset
from vlm.models.global_relational_vlm import (
    GlobalRelationalVLM,
)
from vlm.models.qlora_adapter import (
    attach_qlora_adapter,
)
from vlm.scripts.test_backbone import (
    get_vision_device,
    load_image,
)
from vlm.utils.descriptor_statistics import (
    load_descriptor_statistics,
)
from vlm.utils.first_pass_cache import (
    build_cache_identity,
    load_first_pass_cache,
    save_first_pass_cache,
)


MODEL_NAME = "OpenGVLab/InternVL3-8B"

CACHE_DIR = (
    PROJECT_ROOT
    / "pretrained/huggingface"
)
OFFLOAD_DIR = (
    PROJECT_ROOT
    / "pretrained/offload"
)
STATISTICS_PATH = (
    PROJECT_ROOT
    / "vlm/configs/"
    "gqa_descriptor_statistics_"
    "nf4_qlora_5000_seed2026.json"
)
MANIFEST_PATH = (
    PROJECT_ROOT
    / "datasets/gqa/subsets/"
    "train_debug_100_seed2026.jsonl"
)
GQA_ROOT = (
    PROJECT_ROOT
    / "datasets/gqa"
)
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "vlm/outputs/first_pass_cache/tests"
)


def maximum_error(first, second):
    first = first.detach().cpu().float()
    second = second.detach().cpu().float()

    if first.shape != second.shape:
        return float("inf")

    if first.numel() == 0:
        return 0.0

    return (
        first - second
    ).abs().max().item()


def compare_attention_results(
    original,
    restored,
):
    tensor_fields = (
        "per_layer_head",
        "mean_all_raw",
        "mean_local_raw",
        "mean_thumbnail_raw",
        "all_image_l1",
        "local_image_l1",
    )

    errors = {
        name: maximum_error(
            getattr(original, name),
            getattr(restored, name),
        )
        for name in tensor_fields
    }

    metadata_equal = (
        original.layer_indices
        == restored.layer_indices
        and original.image_positions
        == restored.image_positions
        and original.local_image_positions
        == restored.local_image_positions
        and original.thumbnail_positions
        == restored.thumbnail_positions
        and original.question_start
        == restored.question_start
        and original.question_end
        == restored.question_end
        and original.last_question_token
        == restored.last_question_token
    )

    return errors, metadata_equal


def compare_reconstructed(
    original,
    restored,
):
    tensor_fields = (
        "tile_maps",
        "upsampled_tiles",
        "mosaic",
        "original_map",
    )

    errors = {
        name: maximum_error(
            getattr(original, name),
            getattr(restored, name),
        )
        for name in tensor_fields
    }

    metadata_equal = (
        original.token_grid_size
        == restored.token_grid_size
    )

    return errors, metadata_equal


def build_auxiliary_tokens(
    integrated_model,
    first_pass,
):
    global_parameter = next(
        integrated_model
        .global_encoder
        .parameters()
    )

    graph_parameter = next(
        integrated_model
        .graph_encoder
        .parameters()
    )

    attention_map = (
        first_pass
        .reconstructed_attention
        .original_map
        .to(
            device=global_parameter.device,
            dtype=torch.float32,
        )
    )

    local_attention = (
        first_pass
        .attention_result
        .local_image_l1
        .to(
            device=graph_parameter.device,
            dtype=graph_parameter.dtype,
        )
    )

    attention_profiles = (
        first_pass
        .local_attention_profiles
        .to(
            device=graph_parameter.device,
            dtype=graph_parameter.dtype,
        )
    )

    visual_features = (
        first_pass
        .projected_visual_features
        .to(
            device=graph_parameter.device,
            dtype=graph_parameter.dtype,
        )
    )

    with torch.inference_mode():
        global_output = (
            integrated_model.global_encoder(
                attention_map
            )
        )

        graph_output = (
            integrated_model.graph_encoder(
                local_attention=local_attention,
                attention_profiles=(
                    attention_profiles
                ),
                projected_visual_features=(
                    visual_features
                ),
                tile_grid=first_pass.tile_grid,
                image_size=first_pass.image_size,
            )
        )

    return (
        global_output.global_token.detach().cpu(),
        graph_output.graph_token.detach().cpu(),
    )


def mismatch_is_rejected(
    cache_path,
    identity,
    configuration,
    mismatch_kind,
):
    bad_identity = dict(identity)
    bad_configuration = dict(configuration)

    if mismatch_kind == "identity":
        bad_identity["question_sha256"] = (
            "incorrect-question-hash"
        )
    elif mismatch_kind == "configuration":
        bad_configuration["max_num"] = 999
    else:
        raise ValueError(
            f"Unknown mismatch kind: {mismatch_kind}"
        )

    try:
        load_first_pass_cache(
            input_path=cache_path,
            expected_identity=bad_identity,
            expected_configuration=(
                bad_configuration
            ),
            tile_size=448,
        )
    except ValueError:
        return True

    return False


def main():
    dataset = GQADataset(
        manifest_path=MANIFEST_PATH,
        data_root=GQA_ROOT,
        verify_images=True,
    )

    sample = dataset[0]

    print("First Pass Cache sample:")
    print(
        "  Question ID:",
        sample["question_id"],
    )
    print(
        "  Image ID:",
        sample["image_id"],
    )
    print(
        "  Question:",
        sample["question"],
    )
    print(
        "  Answer:",
        sample["answer"],
    )

    original_image, pixel_values, tile_grid = (
        load_image(
            image_path=Path(
                sample["image_path"]
            ),
            image_size=448,
            max_num=1,
        )
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=(
            torch.bfloat16
        ),
        llm_int8_skip_modules=[
            "vision_model",
            "mlp1",
        ],
    )

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        cache_dir=str(CACHE_DIR),
        use_fast=False,
        local_files_only=True,
    )

    print(
        "Loading NF4 InternVL3-8B..."
    )

    backbone = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        cache_dir=str(CACHE_DIR),
        torch_dtype=torch.bfloat16,
        quantization_config=(
            quantization_config
        ),
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory={
            0: "10GiB",
            "cpu": "48GiB",
        },
        offload_folder=str(OFFLOAD_DIR),
        use_flash_attn=False,
        attn_implementation="eager",
        local_files_only=True,
    ).eval()

    print("Attaching QLoRA Adapter...")

    attach_qlora_adapter(
        backbone=backbone,
        rank=8,
        alpha=16,
        dropout=0.05,
        target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ),
        adapter_name="global_relational",
    )

    integrated_model = GlobalRelationalVLM(
        backbone=backbone,
        tokenizer=tokenizer,
    )

    vision_device = get_vision_device(
        backbone
    )

    integrated_model.global_encoder.to(
        device=vision_device,
        dtype=torch.float32,
    )
    integrated_model.graph_encoder.to(
        device=vision_device,
        dtype=torch.float32,
    )

    statistics = load_descriptor_statistics(
        STATISTICS_PATH
    )

    if (
        statistics["metadata"].get(
            "first_pass_backend"
        )
        != "nf4-qlora"
    ):
        raise RuntimeError(
            "Expected NF4 QLoRA descriptor "
            "statistics."
        )

    integrated_model.global_encoder.set_descriptor_statistics(
        mean=statistics["mean"],
        std=statistics["std"],
    )

    integrated_model.freeze_backbone()
    integrated_model.eval()

    pixel_values = pixel_values.to(
        device=vision_device,
        dtype=torch.bfloat16,
    )

    print(
        "\nComputing live Adapter-OFF "
        "First Pass..."
    )

    live_first_pass = (
        integrated_model.extract_first_pass(
            pixel_values=pixel_values,
            question=sample["question"],
            tile_grid=tile_grid,
            image_size=original_image.size,
        )
    )

    identity = build_cache_identity(
        question_id=sample["question_id"],
        image_id=sample["image_id"],
        question=sample["question"],
    )

    configuration = {
        "model": MODEL_NAME,
        "first_pass_backend": "nf4-qlora",
        "adapter_enabled": False,
        "layer_indices": list(
            integrated_model.layer_indices
        ),
        "image_size": 448,
        "max_num": 1,
        "tokens_per_tile": int(
            backbone.num_image_token
        ),
        "attention_implementation": "eager",
        "local_attention_normalization": (
            "l1"
        ),
    }

    cache_path = (
        OUTPUT_ROOT
        / f"{sample['question_id']}.pt"
    )

    print("Saving cache:", cache_path)

    save_first_pass_cache(
        output_path=cache_path,
        first_pass=live_first_pass,
        identity=identity,
        configuration=configuration,
    )

    print("Loading cache...")

    (
        cached_first_pass,
        cache_information,
    ) = load_first_pass_cache(
        input_path=cache_path,
        expected_identity=identity,
        expected_configuration=configuration,
        tile_size=448,
    )

    attention_errors, attention_metadata = (
        compare_attention_results(
            live_first_pass.attention_result,
            cached_first_pass.attention_result,
        )
    )

    reconstruction_errors, reconstruction_metadata = (
        compare_reconstructed(
            live_first_pass.reconstructed_attention,
            cached_first_pass.reconstructed_attention,
        )
    )

    profile_error = maximum_error(
        live_first_pass.local_attention_profiles,
        cached_first_pass.local_attention_profiles,
    )

    visual_error = maximum_error(
        live_first_pass.projected_visual_features,
        cached_first_pass.projected_visual_features,
    )

    (
        live_global_token,
        live_graph_token,
    ) = build_auxiliary_tokens(
        integrated_model,
        live_first_pass,
    )

    (
        cached_global_token,
        cached_graph_token,
    ) = build_auxiliary_tokens(
        integrated_model,
        cached_first_pass,
    )

    global_token_error = maximum_error(
        live_global_token,
        cached_global_token,
    )
    graph_token_error = maximum_error(
        live_graph_token,
        cached_graph_token,
    )

    raw_payload = torch.load(
        cache_path,
        map_location="cpu",
        weights_only=False,
    )

    cached_state = raw_payload["first_pass"]

    compact_cache = (
        "reconstructed_attention"
        not in cached_state
        and "local_attention_profiles"
        not in cached_state
        and "original_map"
        not in cached_state
    )

    identity_rejected = mismatch_is_rejected(
        cache_path=cache_path,
        identity=identity,
        configuration=configuration,
        mismatch_kind="identity",
    )
    configuration_rejected = (
        mismatch_is_rejected(
            cache_path=cache_path,
            identity=identity,
            configuration=configuration,
            mismatch_kind="configuration",
        )
    )

    cache_mebibytes = (
        cache_information["file_size_bytes"]
        / (1024 ** 2)
    )

    print("\nCache summary:")
    print(
        "  File:",
        cache_information["path"],
    )
    print(
        f"  File size: "
        f"{cache_mebibytes:.3f} MiB"
    )
    print(
        "  Schema version:",
        cache_information["schema_version"],
    )
    print(
        "  Compact cache:",
        compact_cache,
    )
    print(
        "  Visual feature shape:",
        tuple(
            cached_first_pass
            .projected_visual_features
            .shape
        ),
    )
    print(
        "  Visual feature dtype:",
        cached_first_pass
        .projected_visual_features
        .dtype,
    )
    print(
        "  Visual feature device:",
        cached_first_pass
        .projected_visual_features
        .device,
    )

    print("\nAttention errors:")

    for name, error in attention_errors.items():
        print(
            f"  {name}: {error:.10f}"
        )

    print(
        "  profiles:",
        f"{profile_error:.10f}",
    )

    print("\nReconstruction errors:")

    for name, error in (
        reconstruction_errors.items()
    ):
        print(
            f"  {name}: {error:.10f}"
        )

    print("\nFeature and Token errors:")
    print(
        "  Visual features:",
        f"{visual_error:.10f}",
    )
    print(
        "  Global Token:",
        f"{global_token_error:.10f}",
    )
    print(
        "  Graph Token:",
        f"{graph_token_error:.10f}",
    )

    all_attention_exact = all(
        error == 0.0
        for error in attention_errors.values()
    )
    all_reconstruction_exact = all(
        error == 0.0
        for error
        in reconstruction_errors.values()
    )

    checks = {
        "compact_cache": compact_cache,
        "attention_metadata": (
            attention_metadata
        ),
        "attention_exact": (
            all_attention_exact
        ),
        "profiles_exact": (
            profile_error == 0.0
        ),
        "reconstruction_metadata": (
            reconstruction_metadata
        ),
        "reconstruction_exact": (
            all_reconstruction_exact
        ),
        "visual_features_exact": (
            visual_error == 0.0
        ),
        "tile_grid_preserved": (
            live_first_pass.tile_grid
            == cached_first_pass.tile_grid
        ),
        "image_size_preserved": (
            live_first_pass.image_size
            == cached_first_pass.image_size
        ),
        "visual_dtype_preserved": (
            live_first_pass
            .projected_visual_features
            .dtype
            == cached_first_pass
            .projected_visual_features
            .dtype
        ),
        "cache_loaded_on_cpu": (
            cached_first_pass
            .projected_visual_features
            .device.type
            == "cpu"
        ),
        "global_token_equivalent": (
            global_token_error <= 1e-6
        ),
        "graph_token_equivalent": (
            graph_token_error <= 1e-6
        ),
        "identity_mismatch_rejected": (
            identity_rejected
        ),
        "configuration_mismatch_rejected": (
            configuration_rejected
        ),
    }

    print("\nChecks:")

    for name, passed in checks.items():
        print(
            f"  {name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    failed = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    if failed:
        raise RuntimeError(
            "First Pass cache checks failed: "
            + ", ".join(failed)
        )

    print(
        "\nFirst Pass Cache round-trip "
        "test: PASS"
    )
    print(
        "Note: this validates exact cache "
        "restoration, not task performance."
    )


if __name__ == "__main__":
    main()
