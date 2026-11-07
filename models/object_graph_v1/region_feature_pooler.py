import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
from torch import nn


@dataclass
class PooledRegionFeature:
    region_index: int
    normalized_xyxy: Tuple[
        float,
        float,
        float,
        float,
    ]
    local_token_indices: Tuple[int, ...]
    intersection_areas: torch.Tensor
    geometric_weights: torch.Tensor
    attention_weights: torch.Tensor
    coverage_area: float
    attention_mass: float
    attention_density: float
    attention_peak: float
    geometric_visual_feature: torch.Tensor
    attention_visual_feature: torch.Tensor
    pooled_attention_profile: torch.Tensor


@dataclass
class RegionFeaturePoolerOutput:
    region_features: Tuple[
        PooledRegionFeature,
        ...,
    ]
    tile_grid: Tuple[int, int]
    token_grid_size: int
    local_token_count: int
    visual_dimension: int
    attention_profile_shape: Tuple[int, ...]


class RegionFeaturePooler(nn.Module):
    """
    Pool InternVL patch features from normalized
    image regions.

    tile_grid uses InternVL's:
        (columns, rows)

    Local tiles must appear before an optional
    global thumbnail in projected_visual_features.
    """

    def __init__(
        self,
        token_grid_size=16,
        eps=1e-12,
    ):
        super().__init__()

        if token_grid_size <= 0:
            raise ValueError(
                "token_grid_size must be positive."
            )

        if eps <= 0.0:
            raise ValueError(
                "eps must be positive."
            )

        self.token_grid_size = int(
            token_grid_size
        )
        self.eps = float(eps)

    @staticmethod
    def _validate_tile_grid(tile_grid):
        if len(tile_grid) != 2:
            raise ValueError(
                "tile_grid must contain "
                "(columns, rows)."
            )

        columns = int(tile_grid[0])
        rows = int(tile_grid[1])

        if columns <= 0 or rows <= 0:
            raise ValueError(
                "tile_grid values must be positive."
            )

        return columns, rows

    @staticmethod
    def _prepare_boxes(normalized_boxes):
        if isinstance(
            normalized_boxes,
            torch.Tensor,
        ):
            boxes = (
                normalized_boxes
                .detach()
                .to(
                    device="cpu",
                    dtype=torch.float64,
                )
            )

            if boxes.ndim == 1:
                boxes = boxes.unsqueeze(0)

            if (
                boxes.ndim != 2
                or boxes.shape[1] != 4
            ):
                raise ValueError(
                    "normalized_boxes must have "
                    "shape (R, 4)."
                )

            box_values = boxes.tolist()

        else:
            box_values = [
                list(box)
                for box in normalized_boxes
            ]

        if not box_values:
            raise ValueError(
                "At least one box is required."
            )

        prepared = []

        for box_index, box in enumerate(
            box_values
        ):
            if len(box) != 4:
                raise ValueError(
                    f"Box {box_index} must contain "
                    "four values."
                )

            x1, y1, x2, y2 = (
                float(value)
                for value in box
            )

            values = torch.tensor(
                [x1, y1, x2, y2],
                dtype=torch.float64,
            )

            if not bool(
                torch.isfinite(values).all()
            ):
                raise ValueError(
                    f"Box {box_index} contains "
                    "NaN or Inf."
                )

            tolerance = 1e-7

            if (
                x1 < -tolerance
                or y1 < -tolerance
                or x2 > 1.0 + tolerance
                or y2 > 1.0 + tolerance
            ):
                raise ValueError(
                    f"Box {box_index} must lie "
                    "inside [0, 1]."
                )

            x1 = min(max(x1, 0.0), 1.0)
            y1 = min(max(y1, 0.0), 1.0)
            x2 = min(max(x2, 0.0), 1.0)
            y2 = min(max(y2, 0.0), 1.0)

            if x2 <= x1 or y2 <= y1:
                raise ValueError(
                    f"Box {box_index} has "
                    "non-positive area."
                )

            prepared.append(
                (x1, y1, x2, y2)
            )

        return tuple(prepared)

    def _prepare_visual_features(
        self,
        projected_visual_features,
        local_tile_count,
    ):
        tokens_per_tile = (
            self.token_grid_size ** 2
        )

        if (
            projected_visual_features.ndim
            == 3
        ):
            tile_count = int(
                projected_visual_features.shape[0]
            )

            actual_tokens_per_tile = int(
                projected_visual_features.shape[1]
            )

            if (
                actual_tokens_per_tile
                != tokens_per_tile
            ):
                raise ValueError(
                    "Unexpected tokens per tile: "
                    f"{actual_tokens_per_tile}; "
                    f"expected {tokens_per_tile}."
                )

            if tile_count < local_tile_count:
                raise ValueError(
                    "There are fewer visual tiles "
                    "than local tiles."
                )

            local_visual = (
                projected_visual_features[
                    :local_tile_count
                ]
                .reshape(
                    local_tile_count
                    * tokens_per_tile,
                    projected_visual_features
                    .shape[-1],
                )
            )

        elif (
            projected_visual_features.ndim
            == 2
        ):
            local_visual = (
                projected_visual_features
            )

        else:
            raise ValueError(
                "projected_visual_features must "
                "have shape (T, P, D) or (T*P, D)."
            )

        expected_tokens = (
            local_tile_count
            * tokens_per_tile
        )

        if local_visual.shape[0] != (
            expected_tokens
        ):
            raise ValueError(
                "Unexpected local visual token "
                f"count: {local_visual.shape[0]}; "
                f"expected {expected_tokens}."
            )

        if not bool(
            torch.isfinite(local_visual).all()
        ):
            raise ValueError(
                "Visual features contain "
                "NaN or Inf."
            )

        return local_visual

    @staticmethod
    def _validate_local_tensor(
        tensor,
        name,
        expected_token_count,
    ):
        if tensor.ndim < 1:
            raise ValueError(
                f"{name} must have at least "
                "one dimension."
            )

        if tensor.shape[0] != (
            expected_token_count
        ):
            raise ValueError(
                f"Unexpected {name} token count: "
                f"{tensor.shape[0]}; expected "
                f"{expected_token_count}."
            )

        if not bool(
            torch.isfinite(tensor).all()
        ):
            raise ValueError(
                f"{name} contains NaN or Inf."
            )

    def _box_patch_intersections(
        self,
        box,
        columns,
        rows,
    ):
        x1, y1, x2, y2 = box

        mosaic_width = (
            columns * self.token_grid_size
        )
        mosaic_height = (
            rows * self.token_grid_size
        )

        first_column = max(
            0,
            int(math.floor(
                x1 * mosaic_width
            )),
        )
        last_column = min(
            mosaic_width - 1,
            int(math.ceil(
                x2 * mosaic_width
            )) - 1,
        )
        first_row = max(
            0,
            int(math.floor(
                y1 * mosaic_height
            )),
        )
        last_row = min(
            mosaic_height - 1,
            int(math.ceil(
                y2 * mosaic_height
            )) - 1,
        )

        token_indices = []
        intersection_areas = []

        for mosaic_row in range(
            first_row,
            last_row + 1,
        ):
            patch_y1 = (
                mosaic_row / mosaic_height
            )
            patch_y2 = (
                (mosaic_row + 1)
                / mosaic_height
            )

            intersection_height = max(
                0.0,
                min(y2, patch_y2)
                - max(y1, patch_y1),
            )

            if intersection_height <= 0.0:
                continue

            tile_row = (
                mosaic_row
                // self.token_grid_size
            )
            patch_row = (
                mosaic_row
                % self.token_grid_size
            )

            for mosaic_column in range(
                first_column,
                last_column + 1,
            ):
                patch_x1 = (
                    mosaic_column
                    / mosaic_width
                )
                patch_x2 = (
                    (mosaic_column + 1)
                    / mosaic_width
                )

                intersection_width = max(
                    0.0,
                    min(x2, patch_x2)
                    - max(x1, patch_x1),
                )

                if intersection_width <= 0.0:
                    continue

                tile_column = (
                    mosaic_column
                    // self.token_grid_size
                )
                patch_column = (
                    mosaic_column
                    % self.token_grid_size
                )

                tile_index = (
                    tile_row * columns
                    + tile_column
                )

                token_index = (
                    tile_index
                    * self.token_grid_size ** 2
                    + patch_row
                    * self.token_grid_size
                    + patch_column
                )

                token_indices.append(
                    token_index
                )
                intersection_areas.append(
                    intersection_width
                    * intersection_height
                )

        if not token_indices:
            raise RuntimeError(
                "A valid box did not overlap "
                "any local patch."
            )

        return (
            tuple(token_indices),
            torch.tensor(
                intersection_areas,
                dtype=torch.float32,
            ),
        )

    def forward(
        self,
        normalized_boxes: Sequence[
            Sequence[float]
        ],
        local_attention: torch.Tensor,
        projected_visual_features:
            torch.Tensor,
        attention_profiles: torch.Tensor,
        tile_grid: Tuple[int, int],
    ) -> RegionFeaturePoolerOutput:
        columns, rows = (
            self._validate_tile_grid(
                tile_grid
            )
        )

        boxes = self._prepare_boxes(
            normalized_boxes
        )

        local_tile_count = columns * rows

        expected_local_tokens = (
            local_tile_count
            * self.token_grid_size ** 2
        )

        local_attention = (
            local_attention.reshape(-1)
        )

        self._validate_local_tensor(
            local_attention,
            "local_attention",
            expected_local_tokens,
        )
        self._validate_local_tensor(
            attention_profiles,
            "attention_profiles",
            expected_local_tokens,
        )

        if bool(
            (local_attention < 0).any()
        ):
            raise ValueError(
                "local_attention must be "
                "non-negative."
            )

        local_visual = (
            self._prepare_visual_features(
                projected_visual_features,
                local_tile_count,
            )
        )

        patch_area = 1.0 / (
            columns
            * rows
            * self.token_grid_size ** 2
        )

        region_features = []

        for region_index, box in enumerate(
            boxes
        ):
            (
                token_indices,
                intersection_areas_cpu,
            ) = self._box_patch_intersections(
                box=box,
                columns=columns,
                rows=rows,
            )

            visual_indices = torch.tensor(
                token_indices,
                device=local_visual.device,
                dtype=torch.long,
            )

            attention_indices = (
                visual_indices.to(
                    local_attention.device
                )
            )

            profile_indices = (
                visual_indices.to(
                    attention_profiles.device
                )
            )

            visual_areas = (
                intersection_areas_cpu.to(
                    device=local_visual.device,
                    dtype=torch.float32,
                )
            )

            geometric_weights = (
                visual_areas
                / visual_areas.sum().clamp_min(
                    self.eps
                )
            )

            selected_visual = (
                local_visual.index_select(
                    dim=0,
                    index=visual_indices,
                )
                .float()
            )

            geometric_visual_feature = (
                torch.sum(
                    selected_visual
                    * geometric_weights
                    .unsqueeze(-1),
                    dim=0,
                )
            )

            selected_attention = (
                local_attention.index_select(
                    dim=0,
                    index=attention_indices,
                )
                .float()
            )

            coverage_fraction = (
                intersection_areas_cpu
                / patch_area
            ).clamp(
                min=0.0,
                max=1.0,
            )

            covered_attention = (
                selected_attention
                .detach()
                .to("cpu")
                * coverage_fraction
            )

            attention_mass = float(
                covered_attention.sum().item()
            )

            attention_peak = float(
                covered_attention.max().item()
            )

            attention_scores = (
                selected_attention.to(
                    local_visual.device
                )
                * visual_areas
            )

            score_sum = (
                attention_scores.sum()
            )

            if (
                float(
                    score_sum
                    .detach()
                    .cpu()
                    .item()
                )
                > self.eps
            ):
                attention_weights = (
                    attention_scores
                    / score_sum
                )
            else:
                attention_weights = (
                    geometric_weights
                )

            attention_visual_feature = (
                torch.sum(
                    selected_visual
                    * attention_weights
                    .unsqueeze(-1),
                    dim=0,
                )
            )

            selected_profiles = (
                attention_profiles
                .index_select(
                    dim=0,
                    index=profile_indices,
                )
                .float()
            )

            profile_weights = (
                geometric_weights.to(
                    selected_profiles.device
                )
            )

            weight_shape = (
                selected_profiles.shape[0],
                *(
                    1
                    for _ in
                    selected_profiles.shape[1:]
                ),
            )

            pooled_attention_profile = (
                torch.sum(
                    selected_profiles
                    * profile_weights.reshape(
                        weight_shape
                    ),
                    dim=0,
                )
            )

            coverage_area = float(
                intersection_areas_cpu
                .sum()
                .item()
            )

            attention_density = (
                attention_mass
                / max(
                    coverage_area,
                    self.eps,
                )
            )

            region_features.append(
                PooledRegionFeature(
                    region_index=region_index,
                    normalized_xyxy=box,
                    local_token_indices=(
                        token_indices
                    ),
                    intersection_areas=(
                        intersection_areas_cpu
                    ),
                    geometric_weights=(
                        geometric_weights
                    ),
                    attention_weights=(
                        attention_weights
                    ),
                    coverage_area=(
                        coverage_area
                    ),
                    attention_mass=(
                        attention_mass
                    ),
                    attention_density=(
                        attention_density
                    ),
                    attention_peak=(
                        attention_peak
                    ),
                    geometric_visual_feature=(
                        geometric_visual_feature
                    ),
                    attention_visual_feature=(
                        attention_visual_feature
                    ),
                    pooled_attention_profile=(
                        pooled_attention_profile
                    ),
                )
            )

        return RegionFeaturePoolerOutput(
            region_features=tuple(
                region_features
            ),
            tile_grid=(columns, rows),
            token_grid_size=(
                self.token_grid_size
            ),
            local_token_count=(
                expected_local_tokens
            ),
            visual_dimension=int(
                local_visual.shape[-1]
            ),
            attention_profile_shape=tuple(
                int(value)
                for value
                in attention_profiles.shape[1:]
            ),
        )