import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from vlm.data.gqa_dataset import (
    GQADataset,
    normalize_gqa_answer,
)
from vlm.scripts.test_backbone import (
    load_image,
)
from vlm.utils.first_pass_cache import (
    build_cache_identity,
    load_first_pass_cache,
)
from vlm.utils.training_checkpoint import (
    load_training_checkpoint,
)
from vlm.utils.training_model_setup import (
    load_training_model,
)
from vlm.utils.training_runtime import (
    load_cache_manifest,
    load_json,
)


MODEL_NAME = "OpenGVLab/InternVL3-8B"

MODEL_CACHE_DIR = (
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
GQA_ROOT = (
    PROJECT_ROOT
    / "datasets/gqa"
)
VAL_MANIFEST = (
    GQA_ROOT
    / "subsets/val_debug_100.jsonl"
)
VAL_CACHE_ROOT = (
    PROJECT_ROOT
    / "vlm/outputs/first_pass_cache/"
    "nf4_qlora_maxnum1_val_debug100"
)
TRAINING_ROOT = (
    PROJECT_ROOT
    / "vlm/outputs/training/"
    "debug100_full_epoch1"
)
RUN_CONFIG_PATH = (
    TRAINING_ROOT
    / "run_config.json"
)
CHECKPOINT_PATH = (
    TRAINING_ROOT
    / "checkpoints/step_000013.pt"
)


def main():
    dataset = GQADataset(
        manifest_path=VAL_MANIFEST,
        data_root=GQA_ROOT,
        verify_images=True,
    )

    (
        cache_metadata,
        cache_record_by_id,
        _,
    ) = load_cache_manifest(
        cache_root=VAL_CACHE_ROOT,
        manifest_path=VAL_MANIFEST,
        dataset=dataset,
    )

    sample = dataset[0]
    question_id = str(
        sample["question_id"]
    )

    print(
        "Checkpoint generation sample:"
    )
    print(
        "  Question ID:",
        question_id,
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
        "  Reference answer:",
        sample["normalized_answer"],
    )

    components = load_training_model(
        model_name=MODEL_NAME,
        model_cache_dir=(
            MODEL_CACHE_DIR
        ),
        offload_dir=OFFLOAD_DIR,
        statistics_path=(
            STATISTICS_PATH
        ),
        gpu_memory="10GiB",
        cpu_memory="48GiB",
    )

    integrated_model = components[
        "integrated_model"
    ]
    vision_device = components[
        "vision_device"
    ]

    training_configuration = (
        load_json(
            RUN_CONFIG_PATH
        )
    )

    print("\nLoading final checkpoint:")
    print(" ", CHECKPOINT_PATH)

    checkpoint_information = (
        load_training_checkpoint(
            input_path=CHECKPOINT_PATH,
            integrated_model=(
                integrated_model
            ),
            optimizer=None,
            scheduler=None,
            expected_configuration=(
                training_configuration
            ),
            adapter_name=(
                "global_relational"
            ),
            restore_rng=False,
        )
    )

    integrated_model.eval()

    cache_record = (
        cache_record_by_id[
            question_id
        ]
    )

    cache_path = (
        VAL_CACHE_ROOT
        / cache_record["cache_file"]
    )

    identity = build_cache_identity(
        question_id=question_id,
        image_id=sample["image_id"],
        question=sample["question"],
    )

    (
        first_pass,
        cache_information,
    ) = load_first_pass_cache(
        input_path=cache_path,
        expected_identity=identity,
        expected_configuration=(
            cache_metadata[
                "configuration"
            ]
        ),
        tile_size=448,
    )

    (
        original_image,
        pixel_values,
        tile_grid,
    ) = load_image(
        image_path=Path(
            sample["image_path"]
        ),
        image_size=448,
        max_num=1,
    )

    if tuple(
        first_pass.tile_grid
    ) != tuple(tile_grid):
        raise RuntimeError(
            "Cached tile grid does not "
            "match the image."
        )

    if tuple(
        first_pass.image_size
    ) != tuple(
        original_image.size
    ):
        raise RuntimeError(
            "Cached image size does not "
            "match the image."
        )

    pixel_values = pixel_values.to(
        device=vision_device,
        dtype=torch.bfloat16,
    )

    extraction_calls = {
        "count": 0,
    }

    def forbidden_first_pass(
        *args,
        **kwargs,
    ):
        extraction_calls["count"] += 1

        raise RuntimeError(
            "First Pass was recomputed "
            "during cached evaluation."
        )

    integrated_model.extract_first_pass = (
        forbidden_first_pass
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(
        "\nGenerating from cached "
        "First Pass..."
    )

    generation_output = (
        integrated_model.generate(
            pixel_values=pixel_values,
            question=sample["question"],
            tile_grid=tile_grid,
            image_size=(
                original_image.size
            ),
            generation_config={
                "max_new_tokens": 16,
                "do_sample": False,
                "num_beams": 1,
            },
            first_pass=first_pass,
        )
    )

    raw_prediction = (
        generation_output.response
    )
    normalized_prediction = (
        normalize_gqa_answer(
            raw_prediction
        )
    )
    normalized_reference = (
        normalize_gqa_answer(
            sample["answer"]
        )
    )

    exact_match = (
        normalized_prediction
        == normalized_reference
    )

    diagnostics = (
        generation_output
        .second_pass_diagnostics
    )

    global_token = (
        generation_output
        .global_output
        .global_token
    )
    graph_token = (
        generation_output
        .graph_output
        .graph_token
    )

    peak_cuda_memory = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
        if torch.cuda.is_available()
        else 0.0
    )

    print("\nGeneration result:")
    print(
        "  Raw prediction:",
        repr(raw_prediction),
    )
    print(
        "  Normalized prediction:",
        repr(
            normalized_prediction
        ),
    )
    print(
        "  Normalized reference:",
        repr(
            normalized_reference
        ),
    )
    print(
        "  Exact match:",
        exact_match,
    )

    print("\nGeneration diagnostics:")
    print(
        "  First Pass extraction calls:",
        extraction_calls["count"],
    )
    print(
        "  First Pass cached:",
        diagnostics.get(
            "first_pass_cached"
        ),
    )
    print(
        "  QLoRA attached:",
        diagnostics.get(
            "qlora_attached"
        ),
    )
    print(
        "  Adapter enabled:",
        diagnostics.get(
            "adapter_enabled"
        ),
    )
    print(
        "  Input order:",
        diagnostics.get(
            "input_order"
        ),
    )
    print(
        "  Global Token:",
        tuple(global_token.shape),
    )
    print(
        "  Graph Token:",
        tuple(graph_token.shape),
    )
    print(
        "  Checkpoint optimizer step:",
        checkpoint_information[
            "progress"
        ]["optimizer_step"],
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
        "checkpoint_step": (
            checkpoint_information[
                "progress"
            ]["optimizer_step"]
            == 13
        ),
        "first_pass_not_recomputed": (
            extraction_calls["count"]
            == 0
        ),
        "first_pass_cached": (
            diagnostics.get(
                "first_pass_cached"
            )
            is True
        ),
        "qlora_attached": (
            diagnostics.get(
                "qlora_attached"
            )
            is True
        ),
        "adapter_enabled": (
            diagnostics.get(
                "adapter_enabled"
            )
            is True
        ),
        "response_nonempty": bool(
            raw_prediction.strip()
        ),
        "input_order": (
            diagnostics.get(
                "input_order"
            )
            == "question_global_graph_image"
        ),
        "global_token_shape": (
            tuple(global_token.shape)
            == (1, 1, 3584)
        ),
        "graph_token_shape": (
            tuple(graph_token.shape)
            == (1, 1, 3584)
        ),
        "global_token_finite": bool(
            torch.isfinite(
                global_token.float()
            ).all()
        ),
        "graph_token_finite": bool(
            torch.isfinite(
                graph_token.float()
            ).all()
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
            "Checkpoint generation checks "
            "failed: "
            + ", ".join(failed)
        )

    print(
        "\nCheckpoint generation test: PASS"
    )
    print(
        "Note: Exact Match is reported "
        "but is not a test pass condition."
    )


if __name__ == "__main__":
    main()
