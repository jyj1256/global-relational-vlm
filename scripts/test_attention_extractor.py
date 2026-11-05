import argparse
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


VLM_ROOT = Path(__file__).resolve().parents[1]

if str(VLM_ROOT) not in sys.path:
    sys.path.insert(0, str(VLM_ROOT))


from models.attention_extractor import (
    SelectiveAttentionCollector,
)
from scripts.test_backbone import (
    get_vision_device,
    load_image,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract question-conditioned image attention "
            "from selected InternVL3 decoder layers."
        )
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="OpenGVLab/InternVL3-8B",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=(
            "/home/dm-magneto/dm_yunjin/"
            "pretrained/huggingface"
        ),
    )
    parser.add_argument(
        "--offload-dir",
        type=str,
        default=(
            "/home/dm-magneto/dm_yunjin/"
            "vlm/outputs/offload"
        ),
    )
    parser.add_argument(
        "--gpu-memory",
        type=str,
        default="10GiB",
    )
    parser.add_argument(
        "--cpu-memory",
        type=str,
        default="48GiB",
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
        "--layers",
        type=str,
        default="10,11,12,13,14,15,16,17",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    return parser.parse_args()


def parse_layer_indices(text):
    return [
        int(value.strip())
        for value in text.split(",")
        if value.strip()
    ]


def print_attention_statistics(
    result,
    tokens_per_tile,
    tile_grid,
    top_k,
):
    print("\nAttention extraction summary:")
    print(
        "  Selected layers:",
        result.layer_indices,
    )
    print(
        "  Per-layer/head shape:",
        tuple(result.per_layer_head.shape),
    )
    print(
        "  Question token range:",
        (
            result.question_start,
            result.question_end,
        ),
    )
    print(
        "  Last question token:",
        result.last_question_token,
    )
    print(
        "  All image tokens:",
        len(result.image_positions),
    )
    print(
        "  Local image tokens:",
        len(result.local_image_positions),
    )
    print(
        "  Thumbnail tokens:",
        len(result.thumbnail_positions),
    )

    print("\nRaw attention mass:")
    print(
        "  All image mass:",
        f"{result.mean_all_raw.sum().item():.8f}",
    )
    print(
        "  Local image mass:",
        f"{result.mean_local_raw.sum().item():.8f}",
    )
    print(
        "  Thumbnail mass:",
        (
            f"{result.mean_thumbnail_raw.sum().item():.8f}"
            if result.mean_thumbnail_raw.numel() > 0
            else "0.00000000"
        ),
    )

    print("\nL1 normalization checks:")
    print(
        "  All-image L1 sum:",
        f"{result.all_image_l1.sum().item():.8f}",
    )
    print(
        "  Local-image L1 sum:",
        f"{result.local_image_l1.sum().item():.8f}",
    )
    print(
        "  Local NaN:",
        bool(
            torch.isnan(
                result.local_image_l1
            ).any()
        ),
    )
    print(
        "  Local Inf:",
        bool(
            torch.isinf(
                result.local_image_l1
            ).any()
        ),
    )
    print(
        "  Local minimum:",
        f"{result.local_image_l1.min().item():.10f}",
    )
    print(
        "  Local maximum:",
        f"{result.local_image_l1.max().item():.10f}",
    )

    local_count = len(
        result.local_image_positions
    )

    top_k = min(top_k, local_count)

    top_values, top_indices = torch.topk(
        result.local_image_l1,
        k=top_k,
    )

    token_grid_size = int(
        math.sqrt(tokens_per_tile)
    )

    print("\nTop local image tokens:")

    for rank, (
        local_index,
        score,
    ) in enumerate(
        zip(
            top_indices.tolist(),
            top_values.tolist(),
        ),
        start=1,
    ):
        tile_index = (
            local_index // tokens_per_tile
        )
        patch_index = (
            local_index % tokens_per_tile
        )

        tile_row = (
            tile_index // tile_grid[0]
        )
        tile_column = (
            tile_index % tile_grid[0]
        )

        patch_row = (
            patch_index // token_grid_size
        )
        patch_column = (
            patch_index % token_grid_size
        )

        sequence_index = (
            result.local_image_positions[
                local_index
            ]
        )

        print(
            f"  {rank:2d}. "
            f"score={score:.10f}, "
            f"sequence={sequence_index}, "
            f"tile=({tile_row}, {tile_column}), "
            f"patch=({patch_row}, {patch_column})"
        )

    print("\nPer-layer image attention mass:")

    local_token_count = len(
        result.local_image_positions
    )

    for position, layer_index in enumerate(
        result.layer_indices
    ):
        layer_attention = (
            result.per_layer_head[position]
        )

        all_mass = (
            layer_attention
            .sum(dim=-1)
            .mean()
            .item()
        )

        local_mass = (
            layer_attention[
                ...,
                :local_token_count,
            ]
            .sum(dim=-1)
            .mean()
            .item()
        )

        thumbnail_mass = all_mass - local_mass

        print(
            f"  Layer {layer_index:2d}: "
            f"all={all_mass:.8f}, "
            f"local={local_mass:.8f}, "
            f"thumbnail={thumbnail_mass:.8f}"
        )


def main():
    args = parse_args()

    image_path = Path(args.image)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    layer_indices = parse_layer_indices(
        args.layers
    )

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
        use_fast=False,
    )

    print("Loading InternVL3-8B...")

    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory={
            0: args.gpu_memory,
            "cpu": args.cpu_memory,
        },
        offload_folder=args.offload_dir,
        offload_state_dict=True,
        use_flash_attn=False,
        attn_implementation="eager",
    ).eval()

    print(
        "LLM attention implementation:",
        model.language_model.config
        ._attn_implementation,
    )

    _, pixel_values, tile_grid = load_image(
        image_path=image_path,
        image_size=args.image_size,
        max_num=args.max_num,
    )

    vision_device = get_vision_device(model)

    pixel_values = pixel_values.to(
        device=vision_device,
        dtype=torch.bfloat16,
    )

    local_tile_count = (
        tile_grid[0] * tile_grid[1]
    )
    total_tile_count = pixel_values.shape[0]
    tokens_per_tile = model.num_image_token

    collector = SelectiveAttentionCollector(
        model=model,
        tokenizer=tokenizer,
        question=args.question,
        layer_indices=layer_indices,
        total_tile_count=total_tile_count,
        local_tile_count=local_tile_count,
        tokens_per_tile=tokens_per_tile,
    )

    collector.install()

    try:
        with torch.inference_mode():
            response = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=(
                    f"<image>\n{args.question}"
                ),
                generation_config={
                    "max_new_tokens": 8,
                    "do_sample": False,
                    "num_beams": 1,
                },
            )
    finally:
        collector.remove()

    result = collector.build_result()

    print_attention_statistics(
        result=result,
        tokens_per_tile=tokens_per_tile,
        tile_grid=tile_grid,
        top_k=args.top_k,
    )

    print("\nShort generated response:")
    print(response)


if __name__ == "__main__":
    main()