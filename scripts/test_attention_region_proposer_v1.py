import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import (
    Image,
    ImageDraw,
    ImageFont,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from vlm.models.object_graph_v1.attention_region_proposer import (
    AttentionRegionProposer,
)


REGION_COLORS = (
    (255, 40, 40),
    (0, 180, 255),
    (0, 210, 100),
    (255, 170, 0),
    (170, 60, 255),
    (255, 40, 180),
    (0, 220, 220),
    (140, 220, 0),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test connected Attention Region "
            "proposals on the GQA baseball image."
        )
    )

    parser.add_argument(
        "--attention-vector",
        type=Path,
        default=(
            PROJECT_ROOT
            / "vlm/outputs/"
            / "paper_figures/"
            / "baseball_2337002/"
            / "local_attention_vector.npy"
        ),
    )

    parser.add_argument(
        "--image",
        type=Path,
        default=(
            PROJECT_ROOT
            / "datasets/gqa/images/"
            / "2337002.jpg"
        ),
    )

    parser.add_argument(
        "--tile-grid",
        type=int,
        nargs=2,
        default=(1, 1),
        metavar=(
            "COLUMNS",
            "ROWS",
        ),
    )

    parser.add_argument(
        "--token-grid-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--peak-relative-threshold",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--growth-relative-threshold",
        type=float,
        default=0.55,
    )

    parser.add_argument(
        "--max-region-radius",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--min-region-mass",
        type=float,
        default=0.005,
    )

    parser.add_argument(
        "--max-regions",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--output-artifact",
        type=Path,
        default=(
            PROJECT_ROOT
            / "vlm/outputs/"
            / "object_graph_visualizations/"
            / "baseball_2337002/"
            / "attention_regions_test.pt"
        ),
    )

    parser.add_argument(
        "--output-image",
        type=Path,
        default=(
            PROJECT_ROOT
            / "vlm/outputs/"
            / "object_graph_visualizations/"
            / "baseball_2337002/"
            / "attention_regions.png"
        ),
    )

    return parser.parse_args()


def load_font(
    size,
    bold=False,
):
    candidates = (
        (
            "DejaVuSans-Bold.ttf",
            "LiberationSans-Bold.ttf",
        )
        if bold
        else (
            "DejaVuSans.ttf",
            "LiberationSans-Regular.ttf",
        )
    )

    for candidate in candidates:
        try:
            return ImageFont.truetype(
                candidate,
                size=size,
            )
        except OSError:
            continue

    return ImageFont.load_default()


def save_region_visualization(
    image_path,
    output,
    output_path,
):
    image = Image.open(
        image_path
    ).convert("RGBA")

    image_width, image_height = (
        image.size
    )

    grid_height, grid_width = (
        output.probability_grid.shape
    )

    patch_overlay = Image.new(
        "RGBA",
        image.size,
        (0, 0, 0, 0),
    )

    patch_draw = ImageDraw.Draw(
        patch_overlay
    )

    for y in range(grid_height):
        for x in range(grid_width):
            if not bool(
                output.selection_mask[y, x]
            ):
                continue

            x1 = round(
                x
                * image_width
                / grid_width
            )

            y1 = round(
                y
                * image_height
                / grid_height
            )

            x2 = round(
                (x + 1)
                * image_width
                / grid_width
            )

            y2 = round(
                (y + 1)
                * image_height
                / grid_height
            )

            patch_draw.rectangle(
                (x1, y1, x2, y2),
                fill=(
                    255,
                    210,
                    0,
                    55,
                ),
            )

    visualized = Image.alpha_composite(
        image,
        patch_overlay,
    )

    draw = ImageDraw.Draw(
        visualized
    )

    line_width = max(
        2,
        round(
            min(image.size) / 180
        ),
    )

    font = load_font(
        size=max(
            14,
            round(
                min(image.size) / 24
            ),
        ),
        bold=True,
    )

    for region in output.regions:
        color = REGION_COLORS[
            region.region_id
            % len(REGION_COLORS)
        ]

        x1, y1, x2, y2 = (
            region.image_xyxy
        )

        draw.rectangle(
            (x1, y1, x2, y2),
            outline=(
                color[0],
                color[1],
                color[2],
                255,
            ),
            width=line_width,
        )

        label = (
            f"A{region.region_id} "
            f"m={region.attention_mass:.3f} "
            f"p={region.patch_count}"
        )

        text_box = draw.textbbox(
            (x1, y1),
            label,
            font=font,
        )

        text_width = (
            text_box[2]
            - text_box[0]
        )

        text_height = (
            text_box[3]
            - text_box[1]
        )

        label_y = max(
            0,
            y1 - text_height - 6,
        )

        draw.rectangle(
            (
                x1,
                label_y,
                min(
                    image_width,
                    x1
                    + text_width
                    + 8,
                ),
                label_y
                + text_height
                + 6,
            ),
            fill=(
                color[0],
                color[1],
                color[2],
                230,
            ),
        )

        draw.text(
            (
                x1 + 4,
                label_y + 2,
            ),
            label,
            fill=(
                0,
                0,
                0,
                255,
            ),
            font=font,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    visualized.convert("RGB").save(
        output_path,
        format="PNG",
    )


def main():
    args = parse_args()

    if not args.attention_vector.is_file():
        raise FileNotFoundError(
            "Attention vector not found: "
            f"{args.attention_vector}"
        )

    if not args.image.is_file():
        raise FileNotFoundError(
            f"Image not found: {args.image}"
        )

    local_attention = torch.from_numpy(
        np.load(
            args.attention_vector
        )
    ).float()

    with Image.open(args.image) as image:
        image_size = image.size

    proposer = AttentionRegionProposer(
        token_grid_size=(
            args.token_grid_size
        ),
        peak_relative_threshold=(
            args.peak_relative_threshold
        ),
        peak_nms_kernel_size=3,
        growth_relative_threshold=(
            args.growth_relative_threshold
        ),
        max_region_radius=(
            args.max_region_radius
        ),
        smoothing_kernel_size=3,
        min_region_patches=1,
        min_region_mass=(
            args.min_region_mass
        ),
        max_regions=args.max_regions,
        border_margin=1,
        border_penalty=0.50,
    )

    output = proposer(
        local_attention=local_attention,
        tile_grid=tuple(
            args.tile_grid
        ),
        image_size=image_size,
    )

    save_region_visualization(
        image_path=args.image,
        output=output,
        output_path=args.output_image,
    )

    print(
        "Attention Region Proposal summary:"
    )

    print(
        "  Local attention:",
        tuple(local_attention.shape),
    )

    print(
        "  Local attention sum:",
        f"{local_attention.sum().item():.8f}",
    )

    print(
        "  Tile grid:",
        output.tile_grid,
    )

    print(
        "  Image size:",
        output.image_size,
    )

    print(
        "  Probability grid:",
        tuple(
            output.probability_grid.shape
        ),
    )

    print(
        "  Probability sum:",
        f"{output.probability_grid.sum().item():.8f}",
    )

    print(
        "  Smoothed sum:",
        f"{output.smoothed_grid.sum().item():.8f}",
    )

    print(
        "  Selected patches:",
        int(
            output.selection_mask.sum()
        ),
    )

    print(
        "  Selected mass:",
        f"{output.selected_attention_mass:.8f}",
    )

    print(
        "  Retained mass:",
        f"{output.retained_attention_mass:.8f}",
    )

    print(
        "  Region count:",
        len(output.regions),
    )

    print("\nRegions:")

    for region in output.regions:
        print(
            f"  A{region.region_id}: "
            f"score={region.proposal_score:.8f}, "
            f"mass={region.attention_mass:.8f}, "
            f"density={region.attention_density:.8f}, "
            f"peak={region.peak_value:.8f}, "
            f"patches={region.patch_count}, "
            f"grid_box={region.mosaic_bbox}, "
            f"image_box=("
            f"{region.image_xyxy[0]:.1f}, "
            f"{region.image_xyxy[1]:.1f}, "
            f"{region.image_xyxy[2]:.1f}, "
            f"{region.image_xyxy[3]:.1f}), "
            f"center=("
            f"{region.normalized_centroid[0]:.4f}, "
            f"{region.normalized_centroid[1]:.4f})"
        )

    tile_columns, tile_rows = (
        output.tile_grid
    )

    expected_grid_shape = (
        tile_rows
        * output.token_grid_size,
        tile_columns
        * output.token_grid_size,
    )

    total_local_tokens = (
        tile_columns
        * tile_rows
        * output.token_grid_size ** 2
    )

    all_region_indices = [
        index
        for region in output.regions
        for index
        in region.local_token_indices
    ]

    boxes_valid = all(
        (
            region.image_xyxy[2]
            > region.image_xyxy[0]
            and region.image_xyxy[3]
            > region.image_xyxy[1]
            and region.image_xyxy[0]
            >= 0.0
            and region.image_xyxy[1]
            >= 0.0
            and region.image_xyxy[2]
            <= image_size[0] + 1e-6
            and region.image_xyxy[3]
            <= image_size[1] + 1e-6
        )
        for region in output.regions
    )

    checks = {
        "probability_shape": (
            tuple(
                output.probability_grid.shape
            )
            == expected_grid_shape
        ),
        "probability_sum": (
            abs(
                output
                .probability_grid
                .sum()
                .item()
                - 1.0
            )
            < 1e-6
        ),
        "smoothed_sum": (
            abs(
                output
                .smoothed_grid
                .sum()
                .item()
                - 1.0
            )
            < 1e-6
        ),
        "finite": bool(
            torch.isfinite(
                output.probability_grid
            ).all()
            and torch.isfinite(
                output.smoothed_grid
            ).all()
        ),
        "selection_nonempty": bool(
            output.selection_mask.any()
        ),
        "region_count": (
            0
            < len(output.regions)
            <= args.max_regions
        ),
        "positive_region_mass": all(
            region.attention_mass > 0.0
            for region in output.regions
        ),
        "boxes_valid": boxes_valid,
        "token_indices_valid": all(
            0 <= index < total_local_tokens
            for index
            in all_region_indices
        ),
        "components_disjoint": (
            len(all_region_indices)
            == len(
                set(all_region_indices)
            )
        ),
        "retained_not_above_selected": (
            output.retained_attention_mass
            <= (
                output.selected_attention_mass
                + 1e-6
            )
        ),
        "visualization_saved": (
            args.output_image.is_file()
        ),
    }

    print("\nChecks:")

    for name, passed in checks.items():
        print(
            f"  {name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    if not all(checks.values()):
        raise RuntimeError(
            "Attention Region Proposal "
            "test failed."
        )

    args.output_artifact.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "configuration": {
                "token_grid_size": (
                    args.token_grid_size
                ),
                "peak_relative_threshold": (
                    args.peak_relative_threshold
                ),
                "growth_relative_threshold": (
                    args.growth_relative_threshold
                ),
                "max_region_radius": (
                    args.max_region_radius
                ),
                "min_region_mass": (
                    args.min_region_mass
                ),
                "max_regions": (
                    args.max_regions
                ),
            },
            "probability_grid": (
                output.probability_grid
            ),
            "smoothed_grid": (
                output.smoothed_grid
            ),
            "selection_mask": (
                output.selection_mask
            ),
            "regions": output.regions,
            "tile_grid": output.tile_grid,
            "image_size": output.image_size,
            "selected_attention_mass": (
                output
                .selected_attention_mass
            ),
            "retained_attention_mass": (
                output
                .retained_attention_mass
            ),
        },
        args.output_artifact,
    )

    print("\nSaved outputs:")
    print(
        "  Artifact:",
        args.output_artifact,
    )
    print(
        "  Visualization:",
        args.output_image,
    )

    print(
        "\nAttention Region Proposal "
        "test: PASS"
    )

    print(
        "Note: numerical validity passed, "
        "but region quality requires visual "
        "inspection."
    )


if __name__ == "__main__":
    main()