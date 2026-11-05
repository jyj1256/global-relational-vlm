import math
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
    get_trainable_adapter_parameters,
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
GQA_ROOT = (
    PROJECT_ROOT
    / "datasets/gqa"
)
MANIFEST_PATH = (
    GQA_ROOT
    / "subsets/"
    "train_debug_100_seed2026.jsonl"
)
FIRST_PASS_CACHE_ROOT = (
    PROJECT_ROOT
    / "vlm/outputs/first_pass_cache/"
    "nf4_qlora_maxnum1_train_seed2026/"
    "entries"
)
STATISTICS_PATH = (
    PROJECT_ROOT
    / "vlm/configs/"
    "gqa_descriptor_statistics_"
    "nf4_qlora_5000_seed2026.json"
)


def gradient_norm(parameters):
    squared_sum = 0.0

    for parameter in parameters:
        if parameter.grad is None:
            continue

        gradient = (
            parameter.grad
            .detach()
            .float()
        )

        squared_sum += (
            gradient.norm().item() ** 2
        )

    return math.sqrt(squared_sum)


def main():
    dataset = GQADataset(
        manifest_path=MANIFEST_PATH,
        data_root=GQA_ROOT,
        verify_images=True,
    )

    sample = dataset[0]

    print("Cached QLoRA training sample:")
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
        sample["normalized_answer"],
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

    integrated_model.global_encoder.train()
    integrated_model.graph_encoder.train()

    adapter_parameters = (
        get_trainable_adapter_parameters(
            backbone
        )
    )

    global_parameters = [
        parameter
        for parameter
        in integrated_model
        .global_encoder
        .parameters()
        if parameter.requires_grad
    ]

    graph_parameters = [
        parameter
        for parameter
        in integrated_model
        .graph_encoder
        .parameters()
        if parameter.requires_grad
    ]

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

    identity = build_cache_identity(
        question_id=sample["question_id"],
        image_id=sample["image_id"],
        question=sample["question"],
    )

    cache_path = (
        FIRST_PASS_CACHE_ROOT
        / f"{sample['question_id']}.pt"
    )

    print("\nLoading cached First Pass:")
    print(" ", cache_path)

    (
        cached_first_pass,
        cache_information,
    ) = load_first_pass_cache(
        input_path=cache_path,
        expected_identity=identity,
        expected_configuration=configuration,
        tile_size=448,
    )

    if tuple(
        cached_first_pass.tile_grid
    ) != tuple(tile_grid):
        raise RuntimeError(
            "Cached tile grid does not match "
            "the current image."
        )

    if tuple(
        cached_first_pass.image_size
    ) != tuple(original_image.size):
        raise RuntimeError(
            "Cached image size does not match "
            "the current image."
        )

    pixel_values = pixel_values.to(
        device=vision_device,
        dtype=torch.bfloat16,
    )

    extraction_calls = {
        "count": 0,
    }

    def forbidden_first_pass(*args, **kwargs):
        extraction_calls["count"] += 1

        raise RuntimeError(
            "extract_first_pass was called even "
            "though a cache was supplied."
        )

    integrated_model.extract_first_pass = (
        forbidden_first_pass
    )

    trainable_parameters = (
        global_parameters
        + graph_parameters
        + adapter_parameters
    )

    for parameter in trainable_parameters:
        parameter.grad = None

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(
        "\nRunning training forward using "
        "only the cached First Pass..."
    )

    training_output = (
        integrated_model.forward_train(
            pixel_values=pixel_values,
            question=sample["question"],
            answer=(
                sample["normalized_answer"]
            ),
            tile_grid=tile_grid,
            image_size=original_image.size,
            supervise_turn_end=True,
            save_activations_on_cpu=True,
            first_pass=cached_first_pass,
        )
    )

    print("Running backward...")

    training_output.loss.backward()

    global_gradient = gradient_norm(
        global_parameters
    )
    graph_gradient = gradient_norm(
        graph_parameters
    )
    adapter_gradient = gradient_norm(
        adapter_parameters
    )

    adapter_parameter_ids = {
        id(parameter)
        for parameter in adapter_parameters
    }

    base_llm_has_gradient = any(
        parameter.grad is not None
        for parameter
        in backbone.language_model.parameters()
        if id(parameter)
        not in adapter_parameter_ids
    )

    vision_or_mlp1_has_gradient = any(
        parameter.grad is not None
        for module in (
            backbone.vision_model,
            backbone.mlp1,
        )
        for parameter in module.parameters()
    )

    supervised_tokens = int(
        (
            training_output
            .sequence_batch
            .labels
            != -100
        ).sum().item()
    )

    global_token_shape = tuple(
        training_output
        .global_output
        .global_token
        .shape
    )

    graph_token_shape = tuple(
        training_output
        .graph_output
        .graph_token
        .shape
    )

    peak_cuda_memory = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
        if torch.cuda.is_available()
        else 0.0
    )

    total_trainable_count = sum(
        parameter.numel()
        for parameter
        in trainable_parameters
    )

    print("\nCached training summary:")
    print(
        "  Loss:",
        f"{training_output.loss.item():.8f}",
    )
    print(
        "  Supervised tokens:",
        supervised_tokens,
    )
    print(
        "  First Pass extraction calls:",
        extraction_calls["count"],
    )
    print(
        "  Global Token:",
        global_token_shape,
    )
    print(
        "  Graph Token:",
        graph_token_shape,
    )
    print(
        "  Global gradient norm:",
        f"{global_gradient:.8f}",
    )
    print(
        "  Graph gradient norm:",
        f"{graph_gradient:.8f}",
    )
    print(
        "  QLoRA gradient norm:",
        f"{adapter_gradient:.8f}",
    )
    print(
        "  Base LLM has gradient:",
        base_llm_has_gradient,
    )
    print(
        "  Vision/mlp1 has gradient:",
        vision_or_mlp1_has_gradient,
    )
    print(
        "  Trainable parameters:",
        f"{total_trainable_count:,}",
    )
    print(
        "  Cache size:",
        f"{cache_information['file_size_bytes'] / (1024 ** 2):.3f} MiB",
    )
    print(
        "  Peak CUDA memory:",
        f"{peak_cuda_memory:.2f} GiB",
    )

    checks = {
        "cache_file_exists": (
            cache_path.is_file()
        ),
        "descriptor_statistics_fitted": (
            integrated_model
            .global_encoder
            .descriptor_statistics_fitted
        ),
        "first_pass_not_recomputed": (
            extraction_calls["count"] == 0
        ),
        "cached_object_reused": (
            training_output.first_pass
            is cached_first_pass
        ),
        "finite_loss": bool(
            torch.isfinite(
                training_output.loss
            )
        ),
        "positive_loss": (
            training_output.loss.item() > 0
        ),
        "answer_supervision": (
            supervised_tokens > 0
        ),
        "global_token_shape": (
            global_token_shape
            == (1, 1, 3584)
        ),
        "graph_token_shape": (
            graph_token_shape
            == (1, 1, 3584)
        ),
        "global_gradient": (
            global_gradient > 0
        ),
        "graph_gradient": (
            graph_gradient > 0
        ),
        "qlora_gradient": (
            adapter_gradient > 0
        ),
        "base_llm_no_gradient": (
            not base_llm_has_gradient
        ),
        "vision_and_mlp1_no_gradient": (
            not vision_or_mlp1_has_gradient
        ),
        "trainable_parameter_count": (
            total_trainable_count
            == 7_844_251
        ),
        "cached_visual_stays_on_cpu": (
            cached_first_pass
            .projected_visual_features
            .device.type
            == "cpu"
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
            "Cached QLoRA training checks "
            "failed: "
            + ", ".join(failed)
        )

    print(
        "\nCached QLoRA training test: PASS"
    )
    print("Step 24-6 complete.")
    print(
        "Note: this validates cache-aware "
        "gradient flow, not task performance."
    )


if __name__ == "__main__":
    main()