import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from vlm.models.graph_token_injector import (
    chat_with_graph_token,
)
from vlm.scripts.test_backbone import (
    get_vision_device,
    load_image,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test Graph Token injection."
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
        "--graph-token",
        type=Path,
        default=Path(
            "/home/dm-magneto/dm_yunjin/"
            "vlm/outputs/graph_visualizations/candy/"
            "untrained_graph_token_test.pt"
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
    return parser.parse_args()


def main():
    args = parse_args()

    args.offload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact = torch.load(
        args.graph_token,
        map_location="cpu",
        weights_only=True,
    )
    graph_token = artifact["graph_token"]
    metadata = artifact.get("metadata", {})

    print("Graph Token artifact:")
    print(
        f"  Shape: {tuple(graph_token.shape)}"
    )
    print(
        f"  Trained: "
        f"{metadata.get('trained', 'unknown')}"
    )

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir=str(args.cache_dir),
        local_files_only=True,
        use_fast=False,
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
            0: "10GiB",
            "cpu": "48GiB",
        },
        offload_folder=str(args.offload_dir),
        offload_state_dict=True,
        use_flash_attn=False,
        attn_implementation="eager",
    ).eval()

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

    print("\nRunning Graph-only second pass...")

    response, diagnostics = (
        chat_with_graph_token(
            model=model,
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=args.question,
            graph_token=graph_token,
            generation_config={
                "max_new_tokens": 32,
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

    if (
        diagnostics[
            "second_pass_sequence_length"
        ]
        != diagnostics[
            "original_sequence_length"
        ] + 1
    ):
        raise RuntimeError(
            "Exactly one Graph Token was not added."
        )

    if diagnostics["graph_token_count"] != 1:
        raise RuntimeError(
            "Graph Token count is not one."
        )

    if (
        diagnostics["input_order"]
        != "question_graph_image"
    ):
        raise RuntimeError(
            "Unexpected second-pass ordering."
        )

    print("\nGraph Token injection test: PASS")
    print(
        "Note: response quality is not evaluated "
        "because the Graph Token is untrained."
    )


if __name__ == "__main__":
    main()