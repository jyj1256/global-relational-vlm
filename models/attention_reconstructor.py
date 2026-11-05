import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class ReconstructedAttention:
    token_grid_size: int

    # [local_tiles, token_grid, token_grid]
    tile_maps: torch.Tensor

    # [local_tiles, tile_size, tile_size]
    upsampled_tiles: torch.Tensor

    # [tile_rows * tile_size, tile_columns * tile_size]
    mosaic: torch.Tensor

    # [original_height, original_width], sum approximately 1
    original_map: torch.Tensor


def _normalize_sum(values, epsilon=1e-12):
    values = values.clamp_min(0)
    denominator = values.sum().clamp_min(epsilon)
    return values / denominator


def reconstruct_local_attention(
    local_attention: torch.Tensor,
    tile_grid: Tuple[int, int],
    original_size: Tuple[int, int],
    tile_size: int = 448,
):
    """
    Args:
        local_attention:
            One-dimensional local Image Token attention.
            Shape: [local_tile_count * tokens_per_tile]

        tile_grid:
            (columns, rows)

        original_size:
            PIL image size format: (width, height)

        tile_size:
            Spatial size of each InternVL image tile.
    """

    columns, rows = tile_grid
    original_width, original_height = original_size

    local_tile_count = columns * rows

    attention = torch.as_tensor(
        local_attention,
        dtype=torch.float32,
    ).detach().cpu().flatten()

    if attention.numel() % local_tile_count != 0:
        raise ValueError(
            "The local attention length is not divisible "
            "by the number of local tiles."
        )

    tokens_per_tile = (
        attention.numel() // local_tile_count
    )

    token_grid_size = int(
        math.sqrt(tokens_per_tile)
    )

    if token_grid_size**2 != tokens_per_tile:
        raise ValueError(
            "The number of tokens per tile is not "
            "a perfect square."
        )

    attention = _normalize_sum(attention)

    tile_maps = attention.reshape(
        local_tile_count,
        token_grid_size,
        token_grid_size,
    )

    upsampled_tiles = F.interpolate(
        tile_maps.unsqueeze(1),
        size=(tile_size, tile_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    mosaic_height = rows * tile_size
    mosaic_width = columns * tile_size

    mosaic = torch.zeros(
        (mosaic_height, mosaic_width),
        dtype=torch.float32,
    )

    for tile_index in range(local_tile_count):
        tile_row = tile_index // columns
        tile_column = tile_index % columns

        top = tile_row * tile_size
        left = tile_column * tile_size

        mosaic[
            top:top + tile_size,
            left:left + tile_size,
        ] = upsampled_tiles[tile_index]

    original_map = F.interpolate(
        mosaic[
            None,
            None,
            :,
            :,
        ],
        size=(original_height, original_width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    original_map = _normalize_sum(
        original_map
    )

    return ReconstructedAttention(
        token_grid_size=token_grid_size,
        tile_maps=tile_maps,
        upsampled_tiles=upsampled_tiles,
        mosaic=mosaic,
        original_map=original_map,
    )


def _to_display_uint8(
    attention_map,
    percentile=99.5,
):
    array = np.asarray(
        attention_map,
        dtype=np.float32,
    )

    upper = float(
        np.percentile(array, percentile)
    )

    if upper <= 0:
        return np.zeros(
            array.shape,
            dtype=np.uint8,
        )

    normalized = np.clip(
        array / upper,
        0.0,
        1.0,
    )

    return np.round(
        normalized * 255.0
    ).astype(np.uint8)


def save_attention_visualizations(
    original_image: Image.Image,
    reconstructed: ReconstructedAttention,
    local_attention: torch.Tensor,
    tile_grid,
    output_dir,
    alpha=0.45,
    percentile=99.5,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_rgb = np.asarray(
        original_image.convert("RGB"),
        dtype=np.uint8,
    )

    attention_array = (
        reconstructed.original_map
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    display_map = _to_display_uint8(
        attention_array,
        percentile=percentile,
    )

    heatmap_bgr = cv2.applyColorMap(
        display_map,
        cv2.COLORMAP_JET,
    )

    heatmap_rgb = cv2.cvtColor(
        heatmap_bgr,
        cv2.COLOR_BGR2RGB,
    )

    overlay_rgb = np.clip(
        (1.0 - alpha)
        * original_rgb.astype(np.float32)
        + alpha
        * heatmap_rgb.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)

    mosaic_display = _to_display_uint8(
        reconstructed.mosaic.numpy(),
        percentile=percentile,
    )

    Image.fromarray(display_map).save(
        output_dir / "attention_grayscale.png"
    )

    Image.fromarray(heatmap_rgb).save(
        output_dir / "attention_heatmap.png"
    )

    Image.fromarray(overlay_rgb).save(
        output_dir / "attention_overlay.png"
    )

    Image.fromarray(mosaic_display).save(
        output_dir / "attention_mosaic.png"
    )

    np.save(
        output_dir / "attention_map.npy",
        attention_array,
    )

    np.save(
        output_dir / "local_attention_vector.npy",
        torch.as_tensor(
            local_attention,
            dtype=torch.float32,
        )
        .detach()
        .cpu()
        .numpy(),
    )

    metadata = {
        "original_width": original_image.width,
        "original_height": original_image.height,
        "tile_columns": tile_grid[0],
        "tile_rows": tile_grid[1],
        "token_grid_size": (
            reconstructed.token_grid_size
        ),
        "local_token_count": int(
            torch.as_tensor(
                local_attention
            ).numel()
        ),
        "attention_map_sum": float(
            reconstructed.original_map
            .sum()
            .item()
        ),
        "overlay_alpha": alpha,
        "display_percentile": percentile,
    }

    with open(
        output_dir / "metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    return {
        "grayscale": (
            output_dir
            / "attention_grayscale.png"
        ),
        "heatmap": (
            output_dir
            / "attention_heatmap.png"
        ),
        "overlay": (
            output_dir
            / "attention_overlay.png"
        ),
        "mosaic": (
            output_dir
            / "attention_mosaic.png"
        ),
        "attention_array": (
            output_dir
            / "attention_map.npy"
        ),
        "local_vector": (
            output_dir
            / "local_attention_vector.npy"
        ),
        "metadata": (
            output_dir
            / "metadata.json"
        ),
    }