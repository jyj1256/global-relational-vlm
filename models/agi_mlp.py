from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class AGIMLPOutput:
    statistical_feature: torch.Tensor
    standardized_descriptors: torch.Tensor
    group_features: torch.Tensor
    interacted_group_features: torch.Tensor
    mixing_matrix: torch.Tensor
    interaction_gates: torch.Tensor


class DescriptorStandardizer(nn.Module):
    """
    Stores train-split descriptor statistics.

    Before statistics are fitted, this module acts
    as an identity transformation.
    """

    def __init__(
        self,
        descriptor_dimension: int = 16,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.eps = eps

        self.register_buffer(
            "descriptor_mean",
            torch.zeros(descriptor_dimension),
        )

        self.register_buffer(
            "descriptor_std",
            torch.ones(descriptor_dimension),
        )

        self.register_buffer(
            "fitted",
            torch.tensor(False),
        )

    @torch.no_grad()
    def set_statistics(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
    ):
        mean = torch.as_tensor(
            mean,
            dtype=self.descriptor_mean.dtype,
            device=self.descriptor_mean.device,
        )

        std = torch.as_tensor(
            std,
            dtype=self.descriptor_std.dtype,
            device=self.descriptor_std.device,
        )

        if mean.shape != self.descriptor_mean.shape:
            raise ValueError(
                "Unexpected mean shape: "
                f"{tuple(mean.shape)}"
            )

        if std.shape != self.descriptor_std.shape:
            raise ValueError(
                "Unexpected std shape: "
                f"{tuple(std.shape)}"
            )

        if torch.any(std <= 0.0):
            raise ValueError(
                "All standard deviations must be positive."
            )

        self.descriptor_mean.copy_(mean)
        self.descriptor_std.copy_(
            std.clamp_min(self.eps)
        )
        self.fitted.fill_(True)

    def forward(
        self,
        descriptors: torch.Tensor,
    ) -> torch.Tensor:
        return (
            descriptors - self.descriptor_mean
        ) / self.descriptor_std.clamp_min(
            self.eps
        )


class GatedGroupEncoder(nn.Module):
    """
    Gated residual MLP for one 4-D descriptor group.
    """

    def __init__(
        self,
        input_dimension: int = 4,
        hidden_dimension: int = 32,
        expansion_dimension: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.activation_projection = nn.Linear(
            input_dimension,
            expansion_dimension,
        )

        self.value_projection = nn.Linear(
            input_dimension,
            expansion_dimension,
        )

        self.output_projection = nn.Linear(
            expansion_dimension,
            hidden_dimension,
        )

        self.skip_projection = nn.Linear(
            input_dimension,
            hidden_dimension,
        )

        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        group_descriptors: torch.Tensor,
    ) -> torch.Tensor:
        activated = self.activation(
            self.activation_projection(
                group_descriptors
            )
        )

        values = self.value_projection(
            group_descriptors
        )

        gated_features = activated * values

        update = self.output_projection(
            gated_features
        )

        update = self.dropout(update)

        skip = self.skip_projection(
            group_descriptors
        )

        return skip + update


class AdaptiveGroupInteractionMLP(nn.Module):
    """
    Adaptive Group Interaction MLP.

    Input:
        descriptors: (B, 16)

    Groups:
        concentration: descriptors 0:4
        spatial bias: descriptors 4:8
        shape: descriptors 8:12
        frequency: descriptors 12:16

    Output:
        statistical feature: (B, 128)
    """

    def __init__(
        self,
        descriptor_dimension: int = 16,
        group_count: int = 4,
        descriptors_per_group: int = 4,
        group_dimension: int = 32,
        expansion_dimension: int = 64,
        output_dimension: int = 128,
        controller_dimension: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        expected_dimension = (
            group_count * descriptors_per_group
        )

        if descriptor_dimension != expected_dimension:
            raise ValueError(
                "descriptor_dimension must equal "
                "group_count * descriptors_per_group."
            )

        self.descriptor_dimension = (
            descriptor_dimension
        )
        self.group_count = group_count
        self.descriptors_per_group = (
            descriptors_per_group
        )
        self.group_dimension = group_dimension
        self.output_dimension = output_dimension

        self.standardizer = DescriptorStandardizer(
            descriptor_dimension=descriptor_dimension,
        )

        self.group_encoders = nn.ModuleList(
            [
                GatedGroupEncoder(
                    input_dimension=(
                        descriptors_per_group
                    ),
                    hidden_dimension=group_dimension,
                    expansion_dimension=(
                        expansion_dimension
                    ),
                    dropout=dropout,
                )
                for _ in range(group_count)
            ]
        )

        controller_output_dimension = (
            group_count * group_count
            + group_count
        )

        self.controller = nn.Sequential(
            nn.Linear(
                descriptor_dimension,
                controller_dimension,
            ),
            nn.SiLU(),
            nn.Linear(
                controller_dimension,
                controller_output_dimension,
            ),
        )

        flattened_group_dimension = (
            group_count * group_dimension
        )

        self.interaction_projection = nn.Linear(
            flattened_group_dimension,
            output_dimension,
        )

        self.raw_projection = nn.Linear(
            descriptor_dimension,
            output_dimension,
        )

        self.output_norm = nn.LayerNorm(
            output_dimension
        )

        self.output_dropout = nn.Dropout(dropout)

        self._initialize_controller()

    def _initialize_controller(self):
        final_layer = self.controller[-1]

        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

        interaction_logit_count = (
            self.group_count * self.group_count
        )

        with torch.no_grad():
            final_layer.bias[
                interaction_logit_count:
            ].fill_(-2.0)

    def _split_groups(
        self,
        descriptors: torch.Tensor,
    ):
        return torch.split(
            descriptors,
            self.descriptors_per_group,
            dim=1,
        )

    def _build_mixing_matrix(
        self,
        controller_output: torch.Tensor,
    ):
        batch_size = controller_output.shape[0]

        interaction_logit_count = (
            self.group_count * self.group_count
        )

        interaction_logits = controller_output[
            :, :interaction_logit_count
        ]

        interaction_logits = (
            interaction_logits.view(
                batch_size,
                self.group_count,
                self.group_count,
            )
        )

        diagonal_mask = torch.eye(
            self.group_count,
            dtype=torch.bool,
            device=interaction_logits.device,
        )

        interaction_logits = (
            interaction_logits.masked_fill(
                diagonal_mask[None],
                -1e4,
            )
        )

        mixing_matrix = torch.softmax(
            interaction_logits,
            dim=-1,
        )

        gate_logits = controller_output[
            :, interaction_logit_count:
        ]

        interaction_gates = torch.sigmoid(
            gate_logits
        )

        return mixing_matrix, interaction_gates

    def forward(
        self,
        descriptors: torch.Tensor,
    ) -> AGIMLPOutput:
        if descriptors.ndim != 2:
            raise ValueError(
                "descriptors must have shape (B, 16)."
            )

        if descriptors.shape[1] != (
            self.descriptor_dimension
        ):
            raise ValueError(
                "Expected descriptor dimension "
                f"{self.descriptor_dimension}, "
                f"but received {descriptors.shape[1]}."
            )

        descriptors = descriptors.float()

        standardized_descriptors = (
            self.standardizer(descriptors)
        )

        descriptor_groups = self._split_groups(
            standardized_descriptors
        )

        group_features = []

        for encoder, group in zip(
            self.group_encoders,
            descriptor_groups,
        ):
            group_features.append(
                encoder(group)
            )

        group_features = torch.stack(
            group_features,
            dim=1,
        )

        controller_output = self.controller(
            standardized_descriptors
        )

        (
            mixing_matrix,
            interaction_gates,
        ) = self._build_mixing_matrix(
            controller_output
        )

        mixed_group_features = torch.bmm(
            mixing_matrix,
            group_features,
        )

        interacted_group_features = (
            group_features
            + interaction_gates.unsqueeze(-1)
            * mixed_group_features
        )

        flattened_features = (
            interacted_group_features.flatten(1)
        )

        interaction_output = (
            self.interaction_projection(
                flattened_features
            )
        )

        raw_output = self.raw_projection(
            standardized_descriptors
        )

        statistical_feature = self.output_norm(
            interaction_output + raw_output
        )

        statistical_feature = (
            self.output_dropout(
                statistical_feature
            )
        )

        return AGIMLPOutput(
            statistical_feature=statistical_feature,
            standardized_descriptors=(
                standardized_descriptors
            ),
            group_features=group_features,
            interacted_group_features=(
                interacted_group_features
            ),
            mixing_matrix=mixing_matrix,
            interaction_gates=interaction_gates,
        )