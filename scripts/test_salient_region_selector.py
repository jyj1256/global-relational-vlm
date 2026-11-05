import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from vlm.models.salient_region_selector import (
    select_salient_regions,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test Top-K salient region selection."
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
        "--attention",
        type=Path,
        default=Path(
            "/home/dm-magneto/dm_yunjin/"
            "vlm/outputs/attention_maps/candy/"
            "local_attention_vector.npy"
        ),
    )
    parser.add_argument("--tile-rows", type=int, default=2)
    parser.add_argument("--tile-cols", type=int, default=2)
    parser.add_argument("--token-rows", type=int, default=16)
    parser.add_argument("--token-cols", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/home/dm-magneto/dm_yunjin/"
            "vlm/outputs/attention_maps/candy"
        ),
    )
    return parser.parse_args()


def load_font(image_size):
    font_size = max(
        16,
        min(image_size) // 70,
    )

    try:
        return ImageFont.truetype(
            "DejaVuSans-Bold.ttf",
            font_size,
        )
    except OSError:
        return ImageFont.load_default()


def draw_regions(image, selection):
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    font = load_font(overlay.size)
    line_width = max(3, min(overlay.size) // 300)

    for region in selection.regions:
        x0, y0, x1, y1 = region.bbox_xyxy
        box = (
            round(x0),
            round(y0),
            round(x1),
            round(y1),
        )

        color = (
            (255, 40, 40)
            if region.rank <= 5
            else (255, 210, 0)
        )

        draw.rectangle(
            box,
            outline=color,
            width=line_width,
        )

        label = str(region.rank)
        text_box = draw.textbbox(
            (box[0], box[1]),
            label,
            font=font,
        )

        draw.rectangle(
            (
                text_box[0] - 3,
                text_box[1] - 3,
                text_box[2] + 3,
                text_box[3] + 3,
            ),
            fill=color,
        )
        draw.text(
            (box[0], box[1]),
            label,
            fill=(0, 0, 0),
            font=font,
        )

    return overlay


def main():
    args = parse_args()
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    local_attention = np.load(args.attention)

    with Image.open(args.image) as source_image:
        image = source_image.convert("RGB")

    selection = select_salient_regions(
        local_attention=local_attention,
        tile_grid=(
            args.tile_rows,
            args.tile_cols,
        ),
        token_grid=(
            args.token_rows,
            args.token_cols,
        ),
        image_size=image.size,
        top_k=args.top_k,
    )

    weight_sum = sum(
        region.selected_weight
        for region in selection.regions
    )

    print("Salient region selection summary:")
    print(f"  Image size: {selection.image_size}")
    print(f"  Tile grid: {selection.tile_grid}")
    print(f"  Token grid: {selection.token_grid}")
    print(f"  Mosaic shape: {selection.mosaic_shape}")
    print(f"  Top-K: {selection.top_k}")
    print(
        f"  Input attention sum: "
        f"{selection.input_attention_sum:.8f}"
    )
    print(
        f"  Selected attention mass: "
        f"{selection.selected_attention_mass:.8f}"
    )
    print(
        f"  Selected weight sum: "
        f"{weight_sum:.8f}"
    )

    print("\nSelected regions:")

    for region in selection.regions:
        print(
            f"  {region.rank:2d}. "
            f"score={region.attention_score:.10f}, "
            f"weight={region.selected_weight:.6f}, "
            f"local_token={region.local_token_index}, "
            f"tile=({region.tile_row}, {region.tile_col}), "
            f"patch=({region.patch_row}, {region.patch_col}), "
            f"mosaic=({region.mosaic_row}, "
            f"{region.mosaic_col})"
        )

    json_path = (
        args.output_dir
        / f"salient_regions_top{args.top_k}.json"
    )
    overlay_path = (
        args.output_dir
        / f"salient_regions_top{args.top_k}.png"
    )

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            asdict(selection),
            file,
            indent=2,
            ensure_ascii=False,
        )

    overlay = draw_regions(
        image=image,
        selection=selection,
    )
    overlay.save(overlay_path)

    print("\nSaved outputs:")
    print(f"  JSON: {json_path}")
    print(f"  Overlay: {overlay_path}")

    if not np.isclose(weight_sum, 1.0):
        raise RuntimeError(
            "Selected region weights do not sum to 1."
        )

    print("\nSalient region selector test: PASS")


if __name__ == "__main__":
    main()