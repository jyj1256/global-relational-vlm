from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class SalientRegion:
    rank: int
    local_token_index: int
    tile_index: int
    tile_row: int
    tile_col: int
    patch_row: int
    patch_col: int
    mosaic_row: int
    mosaic_col: int
    attention_score: float
    selected_weight: float
    bbox_xyxy: Tuple[float, float, float, float]
    center_xy: Tuple[float, float]
    normalized_center_xy: Tuple[float, float]


@dataclass(frozen=True)
class SalientRegionSelection:
    top_k: int
    tile_grid: Tuple[int, int]
    token_grid: Tuple[int, int]
    mosaic_shape: Tuple[int, int]
    image_size: Tuple[int, int]
    input_attention_sum: float
    selected_attention_mass: float
    regions: List[SalientRegion]


def select_salient_regions(
    local_attention,
    tile_grid,
    image_size,
    token_grid=(16, 16),
    top_k=16,
):
    attention = np.asarray(
        local_attention,
        dtype=np.float64,
    ).reshape(-1)

    tile_rows, tile_cols = tile_grid
    patch_rows, patch_cols = token_grid
    image_width, image_height = image_size

    tokens_per_tile = patch_rows * patch_cols
    expected_tokens = (
        tile_rows
        * tile_cols
        * tokens_per_tile
    )

    if attention.size != expected_tokens:
        raise ValueError(
            f"Expected {expected_tokens} local tokens, "
            f"but received {attention.size}."
        )

    if not np.isfinite(attention).all():
        raise ValueError(
            "Attention contains NaN or Inf."
        )

    if np.any(attention < -1e-12):
        raise ValueError(
            "Attention contains negative values."
        )

    if not 1 <= top_k <= expected_tokens:
        raise ValueError(
            f"top_k must be in [1, {expected_tokens}], "
            f"but received {top_k}."
        )

    attention = np.clip(attention, 0.0, None)
    input_attention_sum = float(attention.sum())

    if input_attention_sum <= 0.0:
        raise ValueError(
            "Attention sum must be positive."
        )

    normalized_attention = (
        attention / input_attention_sum
    )

    token_indices = np.arange(
        expected_tokens,
        dtype=np.int64,
    )

    sorted_indices = np.lexsort(
        (
            token_indices,
            -normalized_attention,
        )
    )
    selected_indices = sorted_indices[:top_k]

    selected_attention_mass = float(
        normalized_attention[selected_indices].sum()
    )

    mosaic_rows = tile_rows * patch_rows
    mosaic_cols = tile_cols * patch_cols

    regions = []

    for rank, token_index in enumerate(
        selected_indices,
        start=1,
    ):
        token_index = int(token_index)

        tile_index = token_index // tokens_per_tile
        patch_index = token_index % tokens_per_tile

        tile_row = tile_index // tile_cols
        tile_col = tile_index % tile_cols

        patch_row = patch_index // patch_cols
        patch_col = patch_index % patch_cols

        mosaic_row = (
            tile_row * patch_rows + patch_row
        )
        mosaic_col = (
            tile_col * patch_cols + patch_col
        )

        x0 = image_width * (
            mosaic_col / mosaic_cols
        )
        y0 = image_height * (
            mosaic_row / mosaic_rows
        )
        x1 = image_width * (
            (mosaic_col + 1) / mosaic_cols
        )
        y1 = image_height * (
            (mosaic_row + 1) / mosaic_rows
        )

        center_x = (x0 + x1) / 2.0
        center_y = (y0 + y1) / 2.0

        attention_score = float(
            normalized_attention[token_index]
        )
        selected_weight = (
            attention_score
            / selected_attention_mass
        )

        regions.append(
            SalientRegion(
                rank=rank,
                local_token_index=token_index,
                tile_index=tile_index,
                tile_row=tile_row,
                tile_col=tile_col,
                patch_row=patch_row,
                patch_col=patch_col,
                mosaic_row=mosaic_row,
                mosaic_col=mosaic_col,
                attention_score=attention_score,
                selected_weight=selected_weight,
                bbox_xyxy=(x0, y0, x1, y1),
                center_xy=(center_x, center_y),
                normalized_center_xy=(
                    center_x / image_width,
                    center_y / image_height,
                ),
            )
        )

    return SalientRegionSelection(
        top_k=top_k,
        tile_grid=(tile_rows, tile_cols),
        token_grid=(patch_rows, patch_cols),
        mosaic_shape=(mosaic_rows, mosaic_cols),
        image_size=(image_width, image_height),
        input_attention_sum=input_attention_sum,
        selected_attention_mass=(
            selected_attention_mass
        ),
        regions=regions,
    )