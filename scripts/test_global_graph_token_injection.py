import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from vlm.models.global_graph_token_injector import (
    chat_with_global_graph_tokens,
)
from vlm.scripts.test_backbone import (
    get_vision_device,
    load_image,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test Global and Graph Token injection."
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
        "--global-token",
        type=Path,
        default=Path(
            "/home/dm-magneto/dm_yunjin/"
            "vlm/outputs/attention_maps/candy/"
            "untrained_global_token_test.pt"
        ),
    )

    parser.add_argument(
        "--graph-token",
        type=Path,
        default=Path(
            "/home/dm-magneto/dm_yunjin/"
            "vlm/outputs/graph_visualizations/"
            "candy/untrained_graph_token_test.pt"
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


def load_artifact_token(
    artifact_path,
    token_key,
):
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {artifact_path}"
        )

    artifact = torch.load(
        artifact_path,
        map_location="cpu",
        weights_only=False,
    )

    if torch.is_tensor(artifact):
        token = artifact
        trained = False

    elif isinstance(artifact, dict):
        if token_key not in artifact:
            raise KeyError(
                f"'{token_key}' is missing from "
                f"{artifact_path}. Available keys: "
                f"{sorted(artifact.keys())}"
            )

        token = artifact[token_key]
        trained = bool(
            artifact.get("trained", False)
        )

    else:
        raise TypeError(
            "Unsupported artifact type: "
            f"{type(artifact).__name__}"
        )

    if not torch.is_tensor(token):
        raise TypeError(
            f"'{token_key}' must be a tensor."
        )

    return token, trained


def main():
    args = parse_args()

    (
        global_token,
        global_token_trained,
    ) = load_artifact_token(
        artifact_path=args.global_token,
        token_key="global_token",
    )

    (
        graph_token,
        graph_token_trained,
    ) = load_artifact_token(
        artifact_path=args.graph_token,
        token_key="graph_token",
    )

    print("Global Token artifact:")
    print("  Shape:", tuple(global_token.shape))
    print("  Trained:", global_token_trained)
    print(
        "  Norm:",
        f"{global_token.float().norm().item():.8f}",
    )

    print("\nGraph Token artifact:")
    print("  Shape:", tuple(graph_token.shape))
    print("  Trained:", graph_token_trained)
    print(
        "  Norm:",
        f"{graph_token.float().norm().item():.8f}",
    )

    args.cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.offload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir=str(args.cache_dir),
        local_files_only=True,
    )

    print("Loading InternVL3-8B...")

    model = AutoModel.from_pretrained(
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

    model.img_context_token_id = (
        tokenizer.convert_tokens_to_ids(
            "<IMG_CONTEXT>"
        )
    )

    (
        original_image,
        pixel_values,
        tile_grid,
    ) = load_image(
        image_path=str(args.image),
        image_size=args.image_size,
        max_num=args.max_num,
    )

    vision_device = get_vision_device(model)

    pixel_values = pixel_values.to(
        device=vision_device,
        dtype=torch.bfloat16,
    )

    print("\nInput information:")
    print("  Original size:", original_image.size)
    print("  Tile grid:", tile_grid)
    print(
        "  Pixel tensor:",
        tuple(pixel_values.shape),
    )
    print("  Question:", args.question)

    print(
        "\nRunning Global + Graph second pass..."
    )

    response, diagnostics = (
        chat_with_global_graph_tokens(
            model=model,
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=args.question,
            global_token=global_token,
            graph_token=graph_token,
            generation_config={
                "max_new_tokens": (
                    args.max_new_tokens
                ),
                "do_sample": False,
                "num_beams": 1,
            },
        )
    )

    print("\nSecond-pass diagnostics:")

    for key, value in diagnostics.items():
        print(f"  {key}: {value}")

    print("\nGenerated response:")
    print(response)

    expected_length = (
        diagnostics["original_sequence_length"]
        + 2
    )

    checks = {
        "sequence_length": (
            diagnostics[
                "second_pass_sequence_length"
            ]
            == expected_length
        ),
        "global_token_count": (
            diagnostics["global_token_count"] == 1
        ),
        "graph_token_count": (
            diagnostics["graph_token_count"] == 1
        ),
        "input_order": (
            diagnostics["input_order"]
            == "question_global_graph_image"
        ),
        "global_before_graph": (
            diagnostics["graph_token_index"]
            == diagnostics["global_token_index"] + 1
        ),
        "graph_before_image": (
            diagnostics["reordered_image_start"]
            == diagnostics["graph_token_index"] + 1
        ),
        "image_context_tokens": (
            diagnostics["image_context_tokens"] > 0
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

    if failed_checks:
        raise RuntimeError(
            "Global/Graph injection test failed: "
            + ", ".join(failed_checks)
        )

    print(
        "\nGlobal + Graph Token injection test: PASS"
    )

    print(
        "Note: response quality is not evaluated "
        "because both auxiliary tokens are untrained."
    )


if __name__ == "__main__":
    main()