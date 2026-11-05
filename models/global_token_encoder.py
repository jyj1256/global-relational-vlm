from dataclasses import dataclass

import torch
import torch.nn as nn

from vlm.models.agi_mlp import (
    AGIMLPOutput,
    AdaptiveGroupInteractionMLP,
)
from vlm.models.attention_descriptors import (
    AttentionDescriptorExtractor,
    AttentionDescriptorResult,
)
from vlm.models.global_feature_fusion import (
    DescriptorConditionedFiLMGatedFusion,
    GlobalFusionOutput,
)
from vlm.models.global_token_projector import (
    GlobalTokenProjectionOutput,
    GlobalTokenProjector,
)
from vlm.models.spatial_attention_encoder import (
    SpatialAttentionEncoder,
)


@dataclass
class GlobalTokenEncoderOutput:
    global_token: torch.Tensor
    spatial_feature: torch.Tensor
    descriptor_output: AttentionDescriptorResult
    statistical_output: AGIMLPOutput
    fusion_output: GlobalFusionOutput
    projection_output: GlobalTokenProjectionOutput


class GlobalTokenEncoder(nn.Module):
    """
    Complete Global Token Encoder.

    Input:
        Question-conditioned attention map

    Output:
        One Global Token with shape (B, 1, 3584)
    """

    def __init__(
        self,
        map_size: int = 32,
        descriptor_dimension: int = 16,
        feature_dimension: int = 128,
        llm_hidden_dimension: int = 3584,
        top_p: float = 0.05,
        peak_threshold: float = 0.50,
        max_peak_count: int = 16,
        dropout: float = 0.1,
        modulation_scale: float = 0.5,
        initial_token_scale: float = 0.35,
    ):
        super().__init__()

        self.map_size = map_size
        self.descriptor_dimension = (
            descriptor_dimension
        )
        self.feature_dimension = feature_dimension
        self.llm_hidden_dimension = (
            llm_hidden_dimension
        )

        self.descriptor_extractor = (
            AttentionDescriptorExtractor(
                map_size=map_size,
                top_p=top_p,
                peak_threshold=peak_threshold,
                max_peak_count=max_peak_count,
            )
        )

        self.spatial_encoder = (
            SpatialAttentionEncoder(
                map_size=map_size,
                output_dimension=feature_dimension,
                dropout=dropout,
            )
        )

        self.statistical_encoder = (
            AdaptiveGroupInteractionMLP(
                descriptor_dimension=(
                    descriptor_dimension
                ),
                group_count=4,
                descriptors_per_group=4,
                group_dimension=32,
                expansion_dimension=64,
                output_dimension=feature_dimension,
                controller_dimension=32,
                dropout=dropout,
            )
        )

        self.fusion = (
            DescriptorConditionedFiLMGatedFusion(
                feature_dimension=feature_dimension,
                film_hidden_dimension=(
                    feature_dimension
                ),
                gate_hidden_dimension=(
                    feature_dimension
                ),
                modulation_scale=modulation_scale,
            )
        )

        self.projector = GlobalTokenProjector(
            input_dimension=feature_dimension,
            llm_hidden_dimension=(
                llm_hidden_dimension
            ),
            initial_token_scale=(
                initial_token_scale
            ),
            rms_norm_eps=1e-6,
        )

    @torch.no_grad()
    def set_descriptor_statistics(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
    ):
        self.statistical_encoder.standardizer.set_statistics(
            mean=mean,
            std=std,
        )

    @property
    def descriptor_statistics_fitted(self) -> bool:
        return bool(
            self.statistical_encoder
            .standardizer
            .fitted
            .item()
        )

    def forward(
        self,
        attention_map: torch.Tensor,
    ) -> GlobalTokenEncoderOutput:
        descriptor_output = (
            self.descriptor_extractor(
                attention_map
            )
        )

        spatial_feature = self.spatial_encoder(
            descriptor_output.probability_map
        )

        statistical_output = (
            self.statistical_encoder(
                descriptor_output.descriptors
            )
        )

        fusion_output = self.fusion(
            spatial_feature=spatial_feature,
            statistical_feature=(
                statistical_output
                .statistical_feature
            ),
        )

        projection_output = self.projector(
            fusion_output.fused_feature
        )

        return GlobalTokenEncoderOutput(
            global_token=(
                projection_output.global_token
            ),
            spatial_feature=spatial_feature,
            descriptor_output=descriptor_output,
            statistical_output=(
                statistical_output
            ),
            fusion_output=fusion_output,
            projection_output=(
                projection_output
            ),
        )