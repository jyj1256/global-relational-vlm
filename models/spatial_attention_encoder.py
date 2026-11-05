from typing import Tuple, Union

import torch
import torch.nn as nn


class DepthwiseSeparableBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        group_count: int,
        stride: int = 2,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=input_channels,
                bias=False,
            ),
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=group_count,
                num_channels=output_channels,
            ),
            nn.SiLU(),
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(inputs)


class SpatialAttentionEncoder(nn.Module):
    """
    Lightweight CNN for a question-conditioned
    attention probability map.

    Input:
        (B, 1, 32, 32)

    Output:
        spatial feature: (B, 128)
    """

    def __init__(
        self,
        map_size: int = 32,
        output_dimension: int = 128,
        dropout: float = 0.1,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.map_size = map_size
        self.output_dimension = output_dimension
        self.eps = eps

        self.stem = nn.Sequential(
            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=4,
                num_channels=16,
            ),
            nn.SiLU(),
        )

        self.block_1 = DepthwiseSeparableBlock(
            input_channels=16,
            output_channels=32,
            group_count=8,
            stride=2,
        )

        self.block_2 = DepthwiseSeparableBlock(
            input_channels=32,
            output_channels=64,
            group_count=8,
            stride=2,
        )

        self.block_3 = DepthwiseSeparableBlock(
            input_channels=64,
            output_channels=128,
            group_count=8,
            stride=2,
        )

        self.spatial_pool = nn.AdaptiveAvgPool2d(
            output_size=(4, 4)
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                128 * 4 * 4,
                output_dimension,
            ),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight,
                    a=5 ** 0.5,
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def prepare_input(
        self,
        probability_map: torch.Tensor,
    ) -> torch.Tensor:
        if probability_map.ndim != 4:
            raise ValueError(
                "probability_map must have shape "
                "(B, 1, H, W)."
            )

        if probability_map.shape[1] != 1:
            raise ValueError(
                "probability_map must have one channel."
            )

        if probability_map.shape[-2:] != (
            self.map_size,
            self.map_size,
        ):
            raise ValueError(
                "Expected spatial size "
                f"({self.map_size}, {self.map_size}), "
                f"but received "
                f"{tuple(probability_map.shape[-2:])}."
            )

        probability_map = torch.nan_to_num(
            probability_map,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        probability_map = probability_map.clamp_min(0.0)

        attention_mass = probability_map.sum(
            dim=(-2, -1),
            keepdim=True,
        )

        if torch.any(attention_mass <= self.eps):
            raise ValueError(
                "A probability map contains zero mass."
            )

        probability_map = (
            probability_map
            / attention_mass.clamp_min(self.eps)
        )

        element_count = (
            probability_map.shape[-2]
            * probability_map.shape[-1]
        )

        # The normalized map has mean 1 after scaling.
        scaled_map = (
            probability_map
            * float(element_count)
        )

        return scaled_map

    def extract_feature_map(
        self,
        probability_map: torch.Tensor,
    ) -> torch.Tensor:
        scaled_map = self.prepare_input(
            probability_map
        )

        features = self.stem(scaled_map)
        features = self.block_1(features)
        features = self.block_2(features)
        features = self.block_3(features)
        features = self.spatial_pool(features)

        return features

    def forward(
        self,
        probability_map: torch.Tensor,
        return_feature_map: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        feature_map = self.extract_feature_map(
            probability_map
        )

        spatial_feature = self.projection(
            feature_map
        )

        if return_feature_map:
            return spatial_feature, feature_map

        return spatial_feature