import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from vlm.models.global_relational_vlm import (
    GlobalRelationalVLM,
)
from vlm.scripts.test_backbone import (
    get_vision_device,
    load_image,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test the complete untrained "
            "Global-Relational Two-Pass VLM."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default="OpenGVLab/InternVL3-8B",
    )

    parser.add_argument(
        "--image",
        type=Path,
        default=Path(
            "/home/dm-magneto/dm_yunjin/"
            "vlm/tests/assets/candy.jpg"
        ),
    )

    parser.add_argument(
        "--question",
        type=str,
        default=(
            "What animal is shown on the candy?"
        ),
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            "/home/dm-magneto/dm_yunjin/"
            "pretrained/huggingface"
        ),
    )

    parser.add_argument(
        "--offload-dir",
        type=Path,
        default=Path(
            "/home/dm-magneto/dm_yunjin/"
            "pretrained/offload"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/dm-magneto/dm_yunjin/"
            "vlm/outputs/predictions/"
            "untrained_global_relational_test.pt"
        ),
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=448,
    )

    parser.add_argument(
        "--max-num",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--gpu-memory",
        type=str,
        default="10GiB",
    )

    parser.add_argument(
        "--cpu-memory",
        type=str,
        default="64GiB",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
    )

    return parser.parse_args()


def load_backbone(args):
    args.cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.offload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir=str(args.cache_dir),
        local_files_only=True,
        use_fast=False,
    )

    print("Loading InternVL3-8B...")

    backbone = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir=str(args.cache_dir),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory={
            0: args.gpu_memory,
            "cpu": args.cpu_memory,
        },
        offload_folder=str(args.offload_dir),
        offload_state_dict=True,
        use_flash_attn=False,
        attn_implementation="eager",
    ).eval()

    backbone.img_context_token_id = (
        tokenizer.convert_tokens_to_ids(
            "<IMG_CONTEXT>"
        )
    )

    return backbone, tokenizer


def main():
    args = parse_args()

    torch.manual_seed(2026)

    if not args.image.exists():
        raise FileNotFoundError(
            f"Image not found: {args.image}"
        )

    backbone, tokenizer = load_backbone(args)

    (
        original_image,
        pixel_values,
        tile_grid,
    ) = load_image(
        image_path=str(args.image),
        image_size=args.image_size,
        max_num=args.max_num,
    )

    vision_device = get_vision_device(backbone)

    pixel_values = pixel_values.to(
        device=vision_device,
        dtype=torch.bfloat16,
    )

    model = GlobalRelationalVLM(
        backbone=backbone,
        tokenizer=tokenizer,
        layer_indices=tuple(range(10, 18)),
        tile_size=args.image_size,
    )

    auxiliary_parameters = (
        model.auxiliary_parameter_count()
    )

    print("\nInput information:")
    print("  Image:", args.image)
    print("  Original size:", original_image.size)
    print("  Tile grid:", tile_grid)
    print(
        "  Pixel tensor:",
        tuple(pixel_values.shape),
    )
    print("  Question:", args.question)

    print("\nIntegrated model:")
    print(
        "  Auxiliary parameters:",
        f"{auxiliary_parameters:,}",
    )
    print(
        "  Descriptor statistics fitted:",
        model.descriptor_statistics_fitted,
    )

    print("\nRunning integrated Two-Pass inference...")

    output = model.generate(
        pixel_values=pixel_values,
        question=args.question,
        tile_grid=tile_grid,
        image_size=original_image.size,
        generation_config={
            "max_new_tokens": (
                args.max_new_tokens
            ),
            "do_sample": False,
            "num_beams": 1,
        },
    )

    first_pass = output.first_pass
    attention_result = (
        first_pass.attention_result
    )
    profiles = (
        first_pass.local_attention_profiles
    )
    reconstructed = (
        first_pass.reconstructed_attention
    )
    visual_features = (
        first_pass.projected_visual_features
    )

    global_token = (
        output.global_output.global_token
    )
    graph_token = (
        output.graph_output.graph_token
    )

    diagnostics = (
        output.second_pass_diagnostics
    )

    local_tile_count = (
        tile_grid[0] * tile_grid[1]
    )

    expected_local_tokens = (
        local_tile_count
        * backbone.num_image_token
    )

    expected_total_tokens = (
        pixel_values.shape[0]
        * backbone.num_image_token
    )

    print("\nFirst-pass Attention:")
    print(
        "  Selected layers:",
        attention_result.layer_indices,
    )
    print(
        "  Per-layer/head shape:",
        tuple(
            attention_result
            .per_layer_head
            .shape
        ),
    )
    print(
        "  Local attention:",
        tuple(
            attention_result
            .local_image_l1
            .shape
        ),
    )
    print(
        "  Local attention sum:",
        f"{attention_result.local_image_l1.sum().item():.8f}",
    )
    print(
        "  Attention profiles:",
        tuple(profiles.shape),
    )

    print("\nAttention reconstruction:")
    print(
        "  Mosaic:",
        tuple(reconstructed.mosaic.shape),
    )
    print(
        "  Original map:",
        tuple(
            reconstructed.original_map.shape
        ),
    )
    print(
        "  Original map sum:",
        f"{reconstructed.original_map.sum().item():.8f}",
    )

    print("\nAuxiliary Token generation:")
    print(
        "  Projected visual features:",
        tuple(visual_features.shape),
    )
    print(
        "  Global Token:",
        tuple(global_token.shape),
    )
    print(
        "  Global Token norm:",
        f"{global_token.float().norm().item():.8f}",
    )
    print(
        "  Graph regions:",
        len(
            output.graph_output
            .selection
            .regions
        ),
    )
    print(
        "  Graph Token:",
        tuple(graph_token.shape),
    )
    print(
        "  Graph Token norm:",
        f"{graph_token.float().norm().item():.8f}",
    )

    print("\nSecond-pass diagnostics:")

    for key, value in diagnostics.items():
        print(f"  {key}: {value}")

    print("\nGenerated response:")
    print(output.response)

    global_nan = bool(
        torch.isnan(global_token).any().item()
    )

    global_inf = bool(
        torch.isinf(global_token).any().item()
    )

    graph_nan = bool(
        torch.isnan(graph_token).any().item()
    )

    graph_inf = bool(
        torch.isinf(graph_token).any().item()
    )

    checks = {
        "auxiliary_parameters": (
            auxiliary_parameters == 2_797_979
        ),
        "layer_count": (
            len(attention_result.layer_indices) == 8
        ),
        "attention_layer_shape": (
            attention_result
            .per_layer_head
            .shape
            == (
                8,
                28,
                expected_total_tokens,
            )
        ),
        "local_attention_shape": (
            attention_result
            .local_image_l1
            .shape
            == (expected_local_tokens,)
        ),
        "local_attention_sum": (
            abs(
                attention_result
                .local_image_l1
                .sum()
                .item()
                - 1.0
            )
            < 1e-5
        ),
        "profile_shape": (
            profiles.shape
            == (
                expected_local_tokens,
                8,
                28,
            )
        ),
        "original_map_shape": (
            reconstructed.original_map.shape
            == (
                original_image.height,
                original_image.width,
            )
        ),
        "original_map_sum": (
            abs(
                reconstructed
                .original_map
                .sum()
                .item()
                - 1.0
            )
            < 1e-5
        ),
        "visual_feature_shape": (
            visual_features.shape
            == (
                pixel_values.shape[0],
                backbone.num_image_token,
                3584,
            )
        ),
        "global_token_shape": (
            global_token.shape
            == (1, 1, 3584)
        ),
        "graph_region_count": (
            len(
                output.graph_output
                .selection
                .regions
            )
            == 16
        ),
        "graph_token_shape": (
            graph_token.shape
            == (1, 1, 3584)
        ),
        "sequence_length": (
            diagnostics[
                "second_pass_sequence_length"
            ]
            == diagnostics[
                "original_sequence_length"
            ]
            + 2
        ),
        "input_order": (
            diagnostics["input_order"]
            == "question_global_graph_image"
        ),
        "finite_global": (
            not global_nan and not global_inf
        ),
        "finite_graph": (
            not graph_nan and not graph_inf
        ),
    }

    failed_checks = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    print("\nChecks:")

    for name, passed in checks.items():
        print(
            f"  {name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "trained": False,
            "descriptor_statistics_fitted": (
                model.descriptor_statistics_fitted
            ),
            "question": args.question,
            "response": output.response,
            "tile_grid": tile_grid,
            "layer_indices": (
                attention_result.layer_indices
            ),
            "global_token": (
                global_token.cpu()
            ),
            "graph_token": (
                graph_token.cpu()
            ),
            "descriptors": (
                output.global_output
                .descriptor_output
                .descriptors
                .cpu()
            ),
            "mixing_matrix": (
                output.global_output
                .statistical_output
                .mixing_matrix
                .cpu()
            ),
            "fusion_gate": (
                output.global_output
                .fusion_output
                .fusion_gate
                .cpu()
            ),
            "graph_edge_weights": (
                output.graph_output
                .weighted_graph
                .normalized_edge_weights
                .cpu()
            ),
            "second_pass_diagnostics": (
                diagnostics
            ),
            "checks": checks,
        },
        args.output,
    )

    print("\nSaved integrated test artifact:")
    print(f"  {args.output}")

    if failed_checks:
        raise RuntimeError(
            "Integrated Two-Pass test failed: "
            + ", ".join(failed_checks)
        )

    print(
        "\nGlobal-Relational VLM integration test: PASS"
    )

    print(
        "Note: response quality is not evaluated "
        "because both auxiliary encoders are untrained."
    )


if __name__ == "__main__":
    main()