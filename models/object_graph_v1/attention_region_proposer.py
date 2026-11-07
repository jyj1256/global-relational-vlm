from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class AttentionRegion:
    region_id: int
    proposal_score: float
    attention_mass: float
    attention_density: float
    peak_value: float
    patch_count: int
    mosaic_bbox: Tuple[int, int, int, int]
    normalized_xyxy: Tuple[
        float,
        float,
        float,
        float,
    ]
    image_xyxy: Tuple[
        float,
        float,
        float,
        float,
    ]
    normalized_centroid: Tuple[
        float,
        float,
    ]
    mosaic_indices: Tuple[int, ...]
    local_token_indices: Tuple[int, ...]


@dataclass(frozen=True)
class AttentionRegionProposalOutput:
    probability_grid: torch.Tensor
    smoothed_grid: torch.Tensor
    selection_mask: torch.Tensor
    regions: Tuple[
        AttentionRegion,
        ...,
    ]
    tile_grid: Tuple[int, int]
    image_size: Tuple[int, int]
    token_grid_size: int
    selected_attention_mass: float
    retained_attention_mass: float


class AttentionRegionProposer(nn.Module):
    def __init__(
        self,
        token_grid_size=16,
        peak_relative_threshold=0.15,
        peak_nms_kernel_size=3,
        growth_relative_threshold=0.55,
        max_region_radius=2,
        smoothing_kernel_size=3,
        min_region_patches=1,
        min_region_mass=0.005,
        max_regions=4,
        border_margin=1,
        border_penalty=0.50,
    ):
        super().__init__()

        self.token_grid_size = int(
            token_grid_size
        )

        self.peak_relative_threshold = float(
            peak_relative_threshold
        )

        self.peak_nms_kernel_size = int(
            peak_nms_kernel_size
        )

        self.growth_relative_threshold = float(
            growth_relative_threshold
        )

        self.max_region_radius = int(
            max_region_radius
        )

        self.smoothing_kernel_size = int(
            smoothing_kernel_size
        )

        self.min_region_patches = int(
            min_region_patches
        )

        self.min_region_mass = float(
            min_region_mass
        )

        self.max_regions = int(
            max_regions
        )

        self.border_margin = int(
            border_margin
        )

        self.border_penalty = float(
            border_penalty
        )

        self._validate_configuration()

    def _validate_configuration(self):
        if self.token_grid_size <= 0:
            raise ValueError(
                "token_grid_size must "
                "be positive."
            )

        if not (
            0.0
            < self.peak_relative_threshold
            <= 1.0
        ):
            raise ValueError(
                "peak_relative_threshold must "
                "be in (0, 1]."
            )

        if (
            self.peak_nms_kernel_size <= 0
            or self.peak_nms_kernel_size
            % 2
            == 0
        ):
            raise ValueError(
                "peak_nms_kernel_size must "
                "be a positive odd integer."
            )

        if not (
            0.0
            < self.growth_relative_threshold
            <= 1.0
        ):
            raise ValueError(
                "growth_relative_threshold must "
                "be in (0, 1]."
            )

        if self.max_region_radius < 0:
            raise ValueError(
                "max_region_radius cannot "
                "be negative."
            )

        if (
            self.smoothing_kernel_size <= 0
            or self.smoothing_kernel_size
            % 2
            == 0
        ):
            raise ValueError(
                "smoothing_kernel_size must "
                "be a positive odd integer."
            )

        if self.min_region_patches <= 0:
            raise ValueError(
                "min_region_patches must "
                "be positive."
            )

        if not (
            0.0
            <= self.min_region_mass
            <= 1.0
        ):
            raise ValueError(
                "min_region_mass must "
                "be in [0, 1]."
            )

        if self.max_regions <= 0:
            raise ValueError(
                "max_regions must be positive."
            )

        if self.border_margin < 0:
            raise ValueError(
                "border_margin cannot "
                "be negative."
            )

        if not (
            0.0
            < self.border_penalty
            <= 1.0
        ):
            raise ValueError(
                "border_penalty must "
                "be in (0, 1]."
            )

    @staticmethod
    def _validate_pair(
        values,
        name,
    ):
        if len(values) != 2:
            raise ValueError(
                f"{name} must contain "
                "two values."
            )

        pair = tuple(
            int(value)
            for value in values
        )

        if (
            pair[0] <= 0
            or pair[1] <= 0
        ):
            raise ValueError(
                f"{name} values must "
                "be positive."
            )

        return pair

    def _prepare_probability_grid(
        self,
        local_attention,
        tile_grid,
    ):
        tile_columns, tile_rows = (
            tile_grid
        )

        local_tile_count = (
            tile_columns * tile_rows
        )

        patches_per_tile = (
            self.token_grid_size ** 2
        )

        expected_count = (
            local_tile_count
            * patches_per_tile
        )

        attention = torch.as_tensor(
            local_attention
        ).detach().to(
            device="cpu",
            dtype=torch.float32,
        ).flatten()

        if (
            attention.numel()
            != expected_count
        ):
            raise ValueError(
                "Unexpected local attention "
                f"length: {attention.numel()}. "
                f"Expected {expected_count}."
            )

        if not torch.isfinite(
            attention
        ).all():
            raise ValueError(
                "local_attention contains "
                "NaN or Inf."
            )

        if float(attention.min()) < -1e-8:
            raise ValueError(
                "local_attention contains "
                "a negative value."
            )

        attention = attention.clamp_min(
            0.0
        )

        attention_sum = attention.sum()

        if float(attention_sum) <= 0.0:
            raise ValueError(
                "local_attention must have "
                "positive mass."
            )

        attention = (
            attention / attention_sum
        )

        tile_maps = attention.view(
            local_tile_count,
            self.token_grid_size,
            self.token_grid_size,
        )

        mosaic_rows = []

        for tile_row in range(tile_rows):
            row_tiles = []

            for tile_column in range(
                tile_columns
            ):
                tile_index = (
                    tile_row
                    * tile_columns
                    + tile_column
                )

                row_tiles.append(
                    tile_maps[tile_index]
                )

            mosaic_rows.append(
                torch.cat(
                    row_tiles,
                    dim=1,
                )
            )

        return torch.cat(
            mosaic_rows,
            dim=0,
        ).contiguous()

    def _smooth_grid(
        self,
        probability_grid,
    ):
        if self.smoothing_kernel_size == 1:
            return probability_grid.clone()

        padding = (
            self.smoothing_kernel_size
            // 2
        )

        batched = probability_grid.view(
            1,
            1,
            probability_grid.shape[0],
            probability_grid.shape[1],
        )

        padded = F.pad(
            batched,
            (
                padding,
                padding,
                padding,
                padding,
            ),
            mode="replicate",
        )

        smoothed = F.avg_pool2d(
            padded,
            kernel_size=(
                self.smoothing_kernel_size
            ),
            stride=1,
        ).squeeze(0).squeeze(0)

        return (
            smoothed
            / smoothed.sum().clamp_min(
                1e-12
            )
        )

    def _border_factor(
        self,
        y,
        x,
        height,
        width,
    ):
        if self.border_margin == 0:
            return 1.0

        near_border = (
            y < self.border_margin
            or x < self.border_margin
            or y >= (
                height
                - self.border_margin
            )
            or x >= (
                width
                - self.border_margin
            )
        )

        return (
            self.border_penalty
            if near_border
            else 1.0
        )

    def _find_peak_candidates(
        self,
        smoothed_grid,
    ):
        height, width = (
            smoothed_grid.shape
        )

        kernel_size = (
            self.peak_nms_kernel_size
        )

        padding = kernel_size // 2

        pooled = F.max_pool2d(
            smoothed_grid.view(
                1,
                1,
                height,
                width,
            ),
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        ).squeeze(0).squeeze(0)

        absolute_threshold = (
            smoothed_grid.max()
            * self.peak_relative_threshold
        )

        peak_mask = (
            smoothed_grid
            >= pooled - 1e-12
        ) & (
            smoothed_grid
            >= absolute_threshold
        )

        coordinates = torch.nonzero(
            peak_mask,
            as_tuple=False,
        )

        candidates = []

        for coordinate in coordinates:
            y = int(coordinate[0].item())
            x = int(coordinate[1].item())

            peak_value = float(
                smoothed_grid[y, x]
            )

            border_factor = (
                self._border_factor(
                    y,
                    x,
                    height,
                    width,
                )
            )

            candidates.append(
                {
                    "y": y,
                    "x": x,
                    "peak_value": (
                        peak_value
                    ),
                    "border_factor": (
                        border_factor
                    ),
                    "ranking_score": (
                        peak_value
                        * border_factor
                    ),
                }
            )

        candidates.sort(
            key=lambda item: (
                -item["ranking_score"],
                -item["peak_value"],
                item["y"],
                item["x"],
            )
        )

        suppression_radius = (
            self.peak_nms_kernel_size
            // 2
        )

        retained = []

        for candidate in candidates:
            too_close = any(
                max(
                    abs(
                        candidate["y"]
                        - previous["y"]
                    ),
                    abs(
                        candidate["x"]
                        - previous["x"]
                    ),
                )
                <= suppression_radius
                for previous in retained
            )

            if too_close:
                continue

            retained.append(candidate)

            if len(retained) >= (
                self.max_regions * 4
            ):
                break

        return retained

    def _grow_from_peak(
        self,
        peak,
        smoothed_grid,
        assigned_mask,
    ):
        height, width = (
            smoothed_grid.shape
        )

        peak_y = peak["y"]
        peak_x = peak["x"]

        if bool(
            assigned_mask[
                peak_y,
                peak_x,
            ]
        ):
            return []

        threshold = (
            peak["peak_value"]
            * self.growth_relative_threshold
        )

        stack = [
            (
                peak_y,
                peak_x,
            )
        ]

        visited = set()
        component = []

        neighbor_offsets = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )

        while stack:
            y, x = stack.pop()

            if (y, x) in visited:
                continue

            visited.add((y, x))

            if not (
                0 <= y < height
                and 0 <= x < width
            ):
                continue

            if bool(
                assigned_mask[y, x]
            ):
                continue

            if (
                abs(y - peak_y)
                > self.max_region_radius
                or abs(x - peak_x)
                > self.max_region_radius
            ):
                continue

            if float(
                smoothed_grid[y, x]
            ) < threshold:
                continue

            component.append((y, x))

            for offset_y, offset_x in (
                neighbor_offsets
            ):
                stack.append(
                    (
                        y + offset_y,
                        x + offset_x,
                    )
                )

        return component

    def _local_token_index(
        self,
        y,
        x,
        tile_grid,
    ):
        tile_columns, _ = tile_grid

        tile_row = (
            y // self.token_grid_size
        )

        tile_column = (
            x // self.token_grid_size
        )

        patch_row = (
            y % self.token_grid_size
        )

        patch_column = (
            x % self.token_grid_size
        )

        tile_index = (
            tile_row * tile_columns
            + tile_column
        )

        return (
            tile_index
            * self.token_grid_size ** 2
            + patch_row
            * self.token_grid_size
            + patch_column
        )

    def _component_record(
        self,
        component,
        probability_grid,
        tile_grid,
        image_size,
        peak_value,
        border_factor,
    ):
        height, width = (
            probability_grid.shape
        )

        image_width, image_height = (
            image_size
        )

        ys = [
            coordinate[0]
            for coordinate in component
        ]

        xs = [
            coordinate[1]
            for coordinate in component
        ]

        values = torch.tensor(
            [
                float(
                    probability_grid[y, x]
                )
                for y, x in component
            ],
            dtype=torch.float32,
        )

        attention_mass = float(
            values.sum()
        )

        patch_count = len(component)

        if (
            patch_count
            < self.min_region_patches
        ):
            return None

        if (
            attention_mass
            < self.min_region_mass
        ):
            return None

        x1 = min(xs)
        y1 = min(ys)
        x2 = max(xs) + 1
        y2 = max(ys) + 1

        normalized_xyxy = (
            x1 / width,
            y1 / height,
            x2 / width,
            y2 / height,
        )

        image_xyxy = (
            normalized_xyxy[0]
            * image_width,
            normalized_xyxy[1]
            * image_height,
            normalized_xyxy[2]
            * image_width,
            normalized_xyxy[3]
            * image_height,
        )

        coordinate_weights = (
            values
            / values.sum().clamp_min(
                1e-12
            )
        )

        centroid_x = sum(
            (
                (x + 0.5) / width
            )
            * float(weight)
            for (_, x), weight
            in zip(
                component,
                coordinate_weights,
            )
        )

        centroid_y = sum(
            (
                (y + 0.5) / height
            )
            * float(weight)
            for (y, _), weight
            in zip(
                component,
                coordinate_weights,
            )
        )

        sorted_component = sorted(
            component
        )

        proposal_score = (
            float(peak_value)
            * attention_mass ** 0.5
            * float(border_factor)
        )

        return {
            "proposal_score": (
                proposal_score
            ),
            "attention_mass": (
                attention_mass
            ),
            "attention_density": (
                attention_mass
                / patch_count
            ),
            "peak_value": float(
                values.max()
            ),
            "patch_count": patch_count,
            "mosaic_bbox": (
                x1,
                y1,
                x2,
                y2,
            ),
            "normalized_xyxy": (
                normalized_xyxy
            ),
            "image_xyxy": image_xyxy,
            "normalized_centroid": (
                centroid_x,
                centroid_y,
            ),
            "mosaic_indices": tuple(
                y * width + x
                for y, x
                in sorted_component
            ),
            "local_token_indices": tuple(
                self._local_token_index(
                    y,
                    x,
                    tile_grid,
                )
                for y, x
                in sorted_component
            ),
        }

    @torch.no_grad()
    def forward(
        self,
        local_attention,
        tile_grid,
        image_size,
    ):
        tile_grid = self._validate_pair(
            tile_grid,
            "tile_grid",
        )

        image_size = self._validate_pair(
            image_size,
            "image_size",
        )

        probability_grid = (
            self._prepare_probability_grid(
                local_attention,
                tile_grid,
            )
        )

        smoothed_grid = (
            self._smooth_grid(
                probability_grid
            )
        )

        peak_candidates = (
            self._find_peak_candidates(
                smoothed_grid
            )
        )

        assigned_mask = torch.zeros_like(
            probability_grid,
            dtype=torch.bool,
        )

        records = []

        for peak in peak_candidates:
            component = (
                self._grow_from_peak(
                    peak=peak,
                    smoothed_grid=(
                        smoothed_grid
                    ),
                    assigned_mask=(
                        assigned_mask
                    ),
                )
            )

            if not component:
                continue

            record = (
                self._component_record(
                    component=component,
                    probability_grid=(
                        probability_grid
                    ),
                    tile_grid=tile_grid,
                    image_size=image_size,
                    peak_value=(
                        peak["peak_value"]
                    ),
                    border_factor=(
                        peak["border_factor"]
                    ),
                )
            )

            if record is None:
                continue

            for y, x in component:
                assigned_mask[y, x] = True

            records.append(record)

            if len(records) >= (
                self.max_regions
            ):
                break

        records.sort(
            key=lambda item: (
                -item["proposal_score"],
                -item["attention_mass"],
                item["mosaic_bbox"],
            )
        )

        regions = tuple(
            AttentionRegion(
                region_id=region_id,
                **record,
            )
            for region_id, record
            in enumerate(records)
        )

        selection_mask = torch.zeros_like(
            probability_grid,
            dtype=torch.bool,
        )

        for region in regions:
            selection_mask.view(-1)[
                list(region.mosaic_indices)
            ] = True

        selected_attention_mass = float(
            probability_grid[
                selection_mask
            ].sum()
        )

        retained_attention_mass = sum(
            region.attention_mass
            for region in regions
        )

        return (
            AttentionRegionProposalOutput(
                probability_grid=(
                    probability_grid
                ),
                smoothed_grid=(
                    smoothed_grid
                ),
                selection_mask=(
                    selection_mask
                ),
                regions=regions,
                tile_grid=tile_grid,
                image_size=image_size,
                token_grid_size=(
                    self.token_grid_size
                ),
                selected_attention_mass=(
                    selected_attention_mass
                ),
                retained_attention_mass=(
                    retained_attention_mass
                ),
            )
        )

