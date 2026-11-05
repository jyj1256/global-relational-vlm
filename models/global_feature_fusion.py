from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GlobalFusionOutput:
    fused_feature: torch.Tensor
    normalized_spatial: torch.Tensor
    normalized_statistical: torch.Tensor
    conditioned_spatial: torch.Tensor
    film_gamma: torch.Tensor
    film_beta: torch.Tensor
    fusion_gate: torch.Tensor
    interaction_feature: torch.Tensor
    pre_normalized_feature: torch.Tensor


def validate_branch_features(
    spatial_feature: torch.Tensor,
    statistical_feature: torch.Tensor,
    feature_dimension: int,
):
    if spatial_feature.ndim != 2:
        raise ValueError(
            "spatial_feature must have shape (B, D)."
        )

    if statistical_feature.ndim != 2:
        raise ValueError(
            "statistical_feature must have shape (B, D)."
        )

    if spatial_feature.shape != (
        statistical_feature.shape
    ):
        raise ValueError(
            "Spatial and statistical feature shapes "
            "must match."
        )

    if spatial_feature.shape[1] != feature_dimension:
        raise ValueError(
            f"Expected feature dimension "
            f"{feature_dimension}, but received "
            f"{spatial_feature.shape[1]}."
        )


class ConcatenationFusion(nn.Module):
    """
    F0 baseline:
        LayerNorm
        Concatenation
        Linear projection
    """

    def __init__(
        self,
        feature_dimension: int = 128,
    ):
        super().__init__()

        self.feature_dimension = feature_dimension

        self.spatial_norm = nn.LayerNorm(
            feature_dimension
        )

        self.statistical_norm = nn.LayerNorm(
            feature_dimension
        )

        self.projection = nn.Linear(
            feature_dimension * 2,
            feature_dimension,
        )

        self.output_norm = nn.LayerNorm(
            feature_dimension
        )

    def forward(
        self,
        spatial_feature: torch.Tensor,
        statistical_feature: torch.Tensor,
    ) -> torch.Tensor:
        validate_branch_features(
            spatial_feature,
            statistical_feature,
            self.feature_dimension,
        )

        normalized_spatial = self.spatial_norm(
            spatial_feature
        )

        normalized_statistical = (
            self.statistical_norm(
                statistical_feature
            )
        )

        concatenated = torch.cat(
            (
                normalized_spatial,
                normalized_statistical,
            ),
            dim=-1,
        )

        fused_feature = self.projection(
            concatenated
        )

        return self.output_norm(fused_feature)


class FeatureWiseGatedFusion(nn.Module):
    """
    F1 baseline:
        LayerNorm
        Feature-wise gate
        Weighted sum
    """

    def __init__(
        self,
        feature_dimension: int = 128,
        gate_hidden_dimension: int = 128,
    ):
        super().__init__()

        self.feature_dimension = feature_dimension

        self.spatial_norm = nn.LayerNorm(
            feature_dimension
        )

        self.statistical_norm = nn.LayerNorm(
            feature_dimension
        )

        self.gate_network = nn.Sequential(
            nn.Linear(
                feature_dimension * 2,
                gate_hidden_dimension,
            ),
            nn.SiLU(),
            nn.Linear(
                gate_hidden_dimension,
                feature_dimension,
            ),
        )

        self.output_norm = nn.LayerNorm(
            feature_dimension
        )

        self._initialize_gate()

    def _initialize_gate(self):
        final_layer = self.gate_network[-1]

        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(
        self,
        spatial_feature: torch.Tensor,
        statistical_feature: torch.Tensor,
    ) -> torch.Tensor:
        validate_branch_features(
            spatial_feature,
            statistical_feature,
            self.feature_dimension,
        )

        normalized_spatial = self.spatial_norm(
            spatial_feature
        )

        normalized_statistical = (
            self.statistical_norm(
                statistical_feature
            )
        )

        gate_input = torch.cat(
            (
                normalized_spatial,
                normalized_statistical,
            ),
            dim=-1,
        )

        fusion_gate = torch.sigmoid(
            self.gate_network(gate_input)
        )

        fused_feature = (
            fusion_gate * normalized_spatial
            + (1.0 - fusion_gate)
            * normalized_statistical
        )

        return self.output_norm(fused_feature)


class DescriptorConditionedFiLMGatedFusion(nn.Module):
    """
    Proposed fusion:
        1. Branch-specific LayerNorm
        2. Statistical-conditioned FiLM
        3. Feature-wise gated fusion
        4. Multiplicative interaction
        5. Final LayerNorm
    """

    def __init__(
        self,
        feature_dimension: int = 128,
        film_hidden_dimension: int = 128,
        gate_hidden_dimension: int = 128,
        modulation_scale: float = 0.5,
    ):
        super().__init__()

        self.feature_dimension = feature_dimension
        self.modulation_scale = modulation_scale

        self.spatial_norm = nn.LayerNorm(
            feature_dimension
        )

        self.statistical_norm = nn.LayerNorm(
            feature_dimension
        )

        self.film_network = nn.Sequential(
            nn.Linear(
                feature_dimension,
                film_hidden_dimension,
            ),
            nn.SiLU(),
            nn.Linear(
                film_hidden_dimension,
                feature_dimension * 2,
            ),
        )

        self.gate_network = nn.Sequential(
            nn.Linear(
                feature_dimension * 2,
                gate_hidden_dimension,
            ),
            nn.SiLU(),
            nn.Linear(
                gate_hidden_dimension,
                feature_dimension,
            ),
        )

        self.interaction_projection = nn.Linear(
            feature_dimension,
            feature_dimension,
        )

        self.output_norm = nn.LayerNorm(
            feature_dimension
        )

        self._initialize_fusion()

    def _initialize_fusion(self):
        film_final_layer = self.film_network[-1]

        nn.init.zeros_(film_final_layer.weight)
        nn.init.zeros_(film_final_layer.bias)

        gate_final_layer = self.gate_network[-1]

        nn.init.zeros_(gate_final_layer.weight)
        nn.init.zeros_(gate_final_layer.bias)

        nn.init.zeros_(
            self.interaction_projection.weight
        )

        nn.init.zeros_(
            self.interaction_projection.bias
        )

    def forward(
        self,
        spatial_feature: torch.Tensor,
        statistical_feature: torch.Tensor,
    ) -> GlobalFusionOutput:
        validate_branch_features(
            spatial_feature,
            statistical_feature,
            self.feature_dimension,
        )

        normalized_spatial = self.spatial_norm(
            spatial_feature
        )

        normalized_statistical = (
            self.statistical_norm(
                statistical_feature
            )
        )

        film_parameters = self.film_network(
            normalized_statistical
        )

        raw_gamma, raw_beta = torch.chunk(
            film_parameters,
            chunks=2,
            dim=-1,
        )

        film_gamma = (
            self.modulation_scale
            * torch.tanh(raw_gamma)
        )

        film_beta = (
            self.modulation_scale
            * torch.tanh(raw_beta)
        )

        conditioned_spatial = (
            (1.0 + film_gamma)
            * normalized_spatial
            + film_beta
        )

        gate_input = torch.cat(
            (
                conditioned_spatial,
                normalized_statistical,
            ),
            dim=-1,
        )

        fusion_gate = torch.sigmoid(
            self.gate_network(gate_input)
        )

        gated_feature = (
            fusion_gate * conditioned_spatial
            + (1.0 - fusion_gate)
            * normalized_statistical
        )

        multiplicative_interaction = (
            conditioned_spatial
            * normalized_statistical
        )

        interaction_feature = (
            self.interaction_projection(
                multiplicative_interaction
            )
        )

        pre_normalized_feature = (
            gated_feature + interaction_feature
        )

        fused_feature = self.output_norm(
            pre_normalized_feature
        )

        return GlobalFusionOutput(
            fused_feature=fused_feature,
            normalized_spatial=normalized_spatial,
            normalized_statistical=(
                normalized_statistical
            ),
            conditioned_spatial=conditioned_spatial,
            film_gamma=film_gamma,
            film_beta=film_beta,
            fusion_gate=fusion_gate,
            interaction_feature=interaction_feature,
            pre_normalized_feature=(
                pre_normalized_feature
            ),
        )