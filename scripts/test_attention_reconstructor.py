import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer


VLM_ROOT = Path(__file__).resolve().parents[1]

if str(VLM_ROOT) not in sys.path:
    sys.path.insert(0, str(VLM_ROOT))


from models.attention_extractor import (
    SelectiveAttentionCollector,
)
from models.attention_profile import (
    extract_local_attention_profiles,
)
from models.attention_reconstructor import (
    reconstruct_local_attention,
    save_attention_visualizations,
)
from scripts.test_backbone import (
    get_vision_device,
    load_image,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct InternVL attention in the "
            "original image coordinate system."
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
        "--output-dir",
        type=str,
        default=(
            "/home/dm-magneto/dm_yunjin/"
            "vlm/outputs/attention_maps/candy"
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
        "--alpha",
        type=float,
        default=0.45,
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.5,
    )

    return parser.parse_args()


def parse_layer_indices(text):
    return [
        int(value.strip())
        for value in text.split(",")
        if value.strip()
    ]


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

    original_image, pixel_values, tile_grid = (
        load_image(
            image_path=image_path,
            image_size=args.image_size,
            max_num=args.max_num,
        )
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
            model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=(
                    f"<image>\n{args.question}"
                ),
                generation_config={
                    "max_new_tokens": 1,
                    "do_sample": False,
                    "num_beams": 1,
                },
            )
    finally:
        collector.remove()

    attention_result = collector.build_result()

    local_attention_profiles = (
        extract_local_attention_profiles(
            attention_result
        )
    )

    profile_output_dir = Path(args.output_dir)
    profile_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    profile_path = (
        profile_output_dir
        / "local_attention_profiles.pt"
    )

    torch.save(
        {
            "profiles": (
                local_attention_profiles
                .detach()
                .cpu()
            ),
            "layer_indices": list(
                attention_result.layer_indices
            ),
            "shape": tuple(
                local_attention_profiles.shape
            ),
            "profile_dimension": (
                local_attention_profiles.shape[1]
                * local_attention_profiles.shape[2]
            ),
        },
        profile_path,
    )

    print(
        "Saved local attention profiles:",
        profile_path,
    )

    reconstructed = reconstruct_local_attention(
        local_attention=(
            attention_result.local_image_l1
        ),
        tile_grid=tile_grid,
        original_size=original_image.size,
        tile_size=args.image_size,
    )

    output_paths = save_attention_visualizations(
        original_image=original_image,
        reconstructed=reconstructed,
        local_attention=(
            attention_result.local_image_l1
        ),
        tile_grid=tile_grid,
        output_dir=args.output_dir,
        alpha=args.alpha,
        percentile=args.percentile,
    )

    print("\nAttention reconstruction summary:")
    print(
        "  Original image size:",
        original_image.size,
    )
    print(
        "  Tile grid:",
        tile_grid,
    )
    print(
        "  Local tiles:",
        local_tile_count,
    )
    print(
        "  Tokens per tile:",
        tokens_per_tile,
    )
    print(
        "  Token grid:",
        (
            reconstructed.token_grid_size,
            reconstructed.token_grid_size,
        ),
    )
    print(
        "  Tile maps shape:",
        tuple(
            reconstructed.tile_maps.shape
        ),
    )
    print(
        "  Mosaic shape:",
        tuple(
            reconstructed.mosaic.shape
        ),
    )
    print(
        "  Original map shape:",
        tuple(
            reconstructed.original_map.shape
        ),
    )
    print(
        "  Original map sum:",
        f"{reconstructed.original_map.sum().item():.8f}",
    )
    print(
        "  NaN:",
        bool(
            torch.isnan(
                reconstructed.original_map
            ).any()
        ),
    )
    print(
        "  Inf:",
        bool(
            torch.isinf(
                reconstructed.original_map
            ).any()
        ),
    )

    print("\nSaved outputs:")

    for name, path in output_paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()