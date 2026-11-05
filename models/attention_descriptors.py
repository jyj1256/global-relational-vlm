from dataclasses import dataclass
from math import ceil, log, sqrt
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


DESCRIPTOR_NAMES: Tuple[str, ...] = (
    # Concentration
    "peak_value",
    "top_5_percent_mass",
    "spatial_entropy",
    "hoyer_sparsity",

    # Spatial bias
    "centroid_x",
    "centroid_y",
    "right_left_balance",
    "bottom_top_balance",

    # Shape
    "spread_x",
    "spread_y",
    "eccentricity",
    "peak_count",

    # Frequency
    "low_frequency_energy",
    "mid_frequency_energy",
    "high_frequency_energy",
    "spectral_entropy",
)


DESCRIPTOR_GROUPS: Dict[str, Tuple[int, int]] = {
    "concentration": (0, 4),
    "spatial_bias": (4, 8),
    "shape": (8, 12),
    "frequency": (12, 16),
}


@dataclass
class AttentionDescriptorResult:
    probability_map: torch.Tensor
    descriptors: torch.Tensor
    peak_count_raw: torch.Tensor


class AttentionDescriptorExtractor(nn.Module):
    """
    Extract 16 deterministic descriptors from an attention map.

    Input shape:
        (H, W)
        (B, H, W)
        (B, 1, H, W)

    Output descriptor shape:
        (B, 16)
    """

    def __init__(
        self,
        map_size: int = 32,
        top_p: float = 0.05,
        peak_threshold: float = 0.50,
        max_peak_count: int = 16,
        eps: float = 1e-8,
    ):
        super().__init__()

        if map_size < 4:
            raise ValueError("map_size must be at least 4.")

        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1].")

        if not 0.0 < peak_threshold <= 1.0:
            raise ValueError(
                "peak_threshold must be in (0, 1]."
            )

        self.map_size = map_size
        self.top_p = top_p
        self.peak_threshold = peak_threshold
        self.max_peak_count = max_peak_count
        self.eps = eps

        coordinates = torch.linspace(
            0.0,
            1.0,
            map_size,
            dtype=torch.float32,
        )

        y_grid, x_grid = torch.meshgrid(
            coordinates,
            coordinates,
            indexing="ij",
        )

        self.register_buffer(
            "x_grid",
            x_grid[None, None],
            persistent=False,
        )

        self.register_buffer(
            "y_grid",
            y_grid[None, None],
            persistent=False,
        )

        gaussian_kernel = torch.tensor(
            [
                [1.0, 2.0, 1.0],
                [2.0, 4.0, 2.0],
                [1.0, 2.0, 1.0],
            ],
            dtype=torch.float32,
        )

        gaussian_kernel = (
            gaussian_kernel
            / gaussian_kernel.sum()
        )

        self.register_buffer(
            "gaussian_kernel",
            gaussian_kernel[None, None],
            persistent=False,
        )

        frequency_y = torch.fft.fftshift(
            torch.fft.fftfreq(map_size)
        )

        frequency_x = torch.fft.fftshift(
            torch.fft.fftfreq(map_size)
        )

        fy, fx = torch.meshgrid(
            frequency_y,
            frequency_x,
            indexing="ij",
        )

        radius = torch.sqrt(fx.square() + fy.square())
        radius = radius / radius.max().clamp_min(eps)

        non_dc = radius > 0.0
        low_mask = non_dc & (radius <= 1.0 / 3.0)
        mid_mask = (
            (radius > 1.0 / 3.0)
            & (radius <= 2.0 / 3.0)
        )
        high_mask = radius > 2.0 / 3.0

        self.register_buffer(
            "non_dc_mask",
            non_dc,
            persistent=False,
        )

        self.register_buffer(
            "low_frequency_mask",
            low_mask,
            persistent=False,
        )

        self.register_buffer(
            "mid_frequency_mask",
            mid_mask,
            persistent=False,
        )

        self.register_buffer(
            "high_frequency_mask",
            high_mask,
            persistent=False,
        )

    def _prepare_probability_map(
        self,
        attention_map: torch.Tensor,
    ) -> torch.Tensor:
        attention_map = torch.as_tensor(
            attention_map,
            dtype=torch.float32,
        )

        if attention_map.ndim == 2:
            attention_map = attention_map[None, None]

        elif attention_map.ndim == 3:
            attention_map = attention_map[:, None]

        elif attention_map.ndim == 4:
            if attention_map.shape[1] != 1:
                raise ValueError(
                    "4-D input must have one channel."
                )

        else:
            raise ValueError(
                "attention_map must have 2, 3, or 4 dimensions."
            )

        attention_map = torch.nan_to_num(
            attention_map,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        attention_map = attention_map.clamp_min(0.0)

        original_mass = attention_map.sum(
            dim=(-2, -1),
            keepdim=True,
        )

        if torch.any(original_mass <= self.eps):
            raise ValueError(
                "Attention map contains a sample "
                "with zero total mass."
            )

        probability_map = attention_map / original_mass

        probability_map = F.interpolate(
            probability_map,
            size=(self.map_size, self.map_size),
            mode="area",
        )

        resized_mass = probability_map.sum(
            dim=(-2, -1),
            keepdim=True,
        )

        probability_map = (
            probability_map
            / resized_mass.clamp_min(self.eps)
        )

        return probability_map

    def _calculate_peak_count(
        self,
        probability_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        kernel = self.gaussian_kernel.to(
            device=probability_map.device,
            dtype=probability_map.dtype,
        )

        padded_map = F.pad(
            probability_map,
            pad=(1, 1, 1, 1),
            mode="reflect",
        )

        smoothed = F.conv2d(
            padded_map,
            kernel,
        )

        padded_smoothed = F.pad(
            smoothed,
            pad=(1, 1, 1, 1),
            mode="replicate",
        )

        local_maximum = F.max_pool2d(
            padded_smoothed,
            kernel_size=3,
            stride=1,
        )

        neighborhood_mean = F.avg_pool2d(
            padded_smoothed,
            kernel_size=3,
            stride=1,
        )

        maximum_value = smoothed.amax(
            dim=(-2, -1),
            keepdim=True,
        )

        threshold = (
            self.peak_threshold
            * maximum_value
        )

        is_local_maximum = (
            smoothed
            >= local_maximum - self.eps
        )

        has_prominence = (
            smoothed
            > neighborhood_mean + self.eps
        )

        passes_threshold = smoothed >= threshold

        peak_mask = (
            is_local_maximum
            & has_prominence
            & passes_threshold
        )

        peak_count_raw = peak_mask.sum(
            dim=(-2, -1)
        ).squeeze(1)

        peak_count_capped = peak_count_raw.clamp(
            max=self.max_peak_count
        )

        peak_count_normalized = (
            peak_count_capped.float()
            / float(self.max_peak_count)
        )

        return (
            peak_count_normalized,
            peak_count_raw,
        )

    def _calculate_frequency_descriptors(
        self,
        probability_map: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        spatial_map = probability_map.squeeze(1)

        spectrum = torch.fft.fft2(
            spatial_map,
            norm="ortho",
        )

        spectrum = torch.fft.fftshift(
            spectrum,
            dim=(-2, -1),
        )

        power = spectrum.abs().square()

        non_dc_mask = self.non_dc_mask.to(
            device=power.device,
        )

        low_mask = self.low_frequency_mask.to(
            device=power.device,
        )

        mid_mask = self.mid_frequency_mask.to(
            device=power.device,
        )

        high_mask = self.high_frequency_mask.to(
            device=power.device,
        )

        non_dc_power = power * non_dc_mask
        total_power = non_dc_power.sum(
            dim=(-2, -1)
        )

        safe_total = total_power.clamp_min(self.eps)

        low_energy = (
            (power * low_mask).sum(dim=(-2, -1))
            / safe_total
        )

        mid_energy = (
            (power * mid_mask).sum(dim=(-2, -1))
            / safe_total
        )

        high_energy = (
            (power * high_mask).sum(dim=(-2, -1))
            / safe_total
        )

        spectral_probability = (
            non_dc_power
            / safe_total[:, None, None]
        )

        spectral_entropy = -(
            spectral_probability
            * torch.log(
                spectral_probability + self.eps
            )
        ).sum(dim=(-2, -1))

        frequency_count = int(non_dc_mask.sum().item())

        spectral_entropy = (
            spectral_entropy
            / log(float(frequency_count))
        )

        has_frequency_energy = total_power > self.eps

        low_energy = torch.where(
            has_frequency_energy,
            low_energy,
            torch.zeros_like(low_energy),
        )

        mid_energy = torch.where(
            has_frequency_energy,
            mid_energy,
            torch.zeros_like(mid_energy),
        )

        high_energy = torch.where(
            has_frequency_energy,
            high_energy,
            torch.zeros_like(high_energy),
        )

        spectral_entropy = torch.where(
            has_frequency_energy,
            spectral_entropy,
            torch.zeros_like(spectral_entropy),
        )

        return (
            low_energy,
            mid_energy,
            high_energy,
            spectral_entropy,
        )

    def forward(
        self,
        attention_map: torch.Tensor,
    ) -> AttentionDescriptorResult:
        probability_map = (
            self._prepare_probability_map(
                attention_map
            )
        )

        batch_size = probability_map.shape[0]
        element_count = self.map_size ** 2

        flattened = probability_map.flatten(1)

        peak_value = flattened.max(dim=1).values

        top_k = max(
            1,
            ceil(element_count * self.top_p),
        )

        top_mass = torch.topk(
            flattened,
            k=top_k,
            dim=1,
        ).values.sum(dim=1)

        spatial_entropy = -(
            flattened
            * torch.log(flattened + self.eps)
        ).sum(dim=1)

        spatial_entropy = (
            spatial_entropy / log(float(element_count))
        )

        l1_norm = flattened.sum(dim=1)
        l2_norm = torch.linalg.vector_norm(
            flattened,
            ord=2,
            dim=1,
        )

        square_root_n = sqrt(float(element_count))

        hoyer_sparsity = (
            square_root_n
            - l1_norm / l2_norm.clamp_min(self.eps)
        ) / (square_root_n - 1.0)

        hoyer_sparsity = hoyer_sparsity.clamp(
            min=0.0,
            max=1.0,
        )

        x_grid = self.x_grid.to(
            device=probability_map.device,
            dtype=probability_map.dtype,
        )

        y_grid = self.y_grid.to(
            device=probability_map.device,
            dtype=probability_map.dtype,
        )

        centroid_x = (
            probability_map * x_grid
        ).sum(dim=(-2, -1)).squeeze(1)

        centroid_y = (
            probability_map * y_grid
        ).sum(dim=(-2, -1)).squeeze(1)

        left_mass = (
            probability_map
            * (x_grid < 0.5)
        ).sum(dim=(-2, -1)).squeeze(1)

        right_mass = (
            probability_map
            * (x_grid >= 0.5)
        ).sum(dim=(-2, -1)).squeeze(1)

        top_mass_spatial = (
            probability_map
            * (y_grid < 0.5)
        ).sum(dim=(-2, -1)).squeeze(1)

        bottom_mass = (
            probability_map
            * (y_grid >= 0.5)
        ).sum(dim=(-2, -1)).squeeze(1)

        right_left_balance = right_mass - left_mass
        bottom_top_balance = (
            bottom_mass - top_mass_spatial
        )

        centered_x = (
            x_grid
            - centroid_x[:, None, None, None]
        )

        centered_y = (
            y_grid
            - centroid_y[:, None, None, None]
        )

        variance_x = (
            probability_map
            * centered_x.square()
        ).sum(dim=(-2, -1)).squeeze(1)

        variance_y = (
            probability_map
            * centered_y.square()
        ).sum(dim=(-2, -1)).squeeze(1)

        covariance_xy = (
            probability_map
            * centered_x
            * centered_y
        ).sum(dim=(-2, -1)).squeeze(1)

        spread_x = torch.sqrt(
            variance_x.clamp_min(0.0)
        )

        spread_y = torch.sqrt(
            variance_y.clamp_min(0.0)
        )

        half_difference = (
            variance_x - variance_y
        ) / 2.0

        eigen_offset = torch.sqrt(
            half_difference.square()
            + covariance_xy.square()
        )

        eigen_center = (
            variance_x + variance_y
        ) / 2.0

        lambda_max = (
            eigen_center + eigen_offset
        ).clamp_min(0.0)

        lambda_min = (
            eigen_center - eigen_offset
        ).clamp_min(0.0)

        eccentricity = torch.where(
            lambda_max > self.eps,
            torch.sqrt(
                (
                    1.0
                    - lambda_min
                    / lambda_max.clamp_min(self.eps)
                ).clamp(min=0.0, max=1.0)
            ),
            torch.zeros_like(lambda_max),
        )

        (
            peak_count,
            peak_count_raw,
        ) = self._calculate_peak_count(
            probability_map
        )

        (
            low_frequency_energy,
            mid_frequency_energy,
            high_frequency_energy,
            spectral_entropy,
        ) = self._calculate_frequency_descriptors(
            probability_map
        )

        descriptors = torch.stack(
            (
                peak_value,
                top_mass,
                spatial_entropy,
                hoyer_sparsity,
                centroid_x,
                centroid_y,
                right_left_balance,
                bottom_top_balance,
                spread_x,
                spread_y,
                eccentricity,
                peak_count,
                low_frequency_energy,
                mid_frequency_energy,
                high_frequency_energy,
                spectral_entropy,
            ),
            dim=1,
        )

        if descriptors.shape != (batch_size, 16):
            raise RuntimeError(
                "Unexpected descriptor shape: "
                f"{tuple(descriptors.shape)}"
            )

        return AttentionDescriptorResult(
            probability_map=probability_map,
            descriptors=descriptors,
            peak_count_raw=peak_count_raw,
        )