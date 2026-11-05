from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


@dataclass
class WeightedGNNOutput:
    layer_states: Tuple[torch.Tensor, ...]
    final_node_features: torch.Tensor


class WeightedGraphLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        use_layer_norm=True,
    ):
        super().__init__()

        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(input_dim)
        else:
            self.layer_norm = nn.Identity()

        self.linear = nn.Linear(
            input_dim,
            output_dim,
            bias=False,
        )
        self.activation = nn.GELU()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(
            self.linear.weight
        )

        if isinstance(
            self.layer_norm,
            nn.LayerNorm,
        ):
            nn.init.ones_(
                self.layer_norm.weight
            )
            nn.init.zeros_(
                self.layer_norm.bias
            )

    def forward(
        self,
        node_features,
        normalized_edge_weights,
    ):
        if node_features.ndim != 2:
            raise ValueError(
                "node_features must have shape [K, D]."
            )

        node_count = node_features.shape[0]

        if normalized_edge_weights.shape != (
            node_count,
            node_count,
        ):
            raise ValueError(
                "Edge matrix must have shape [K, K]."
            )

        normalized_features = self.layer_norm(
            node_features
        )
        transformed_features = self.linear(
            normalized_features
        )

        messages = (
            normalized_edge_weights
            @ transformed_features
        )
        updated_features = self.activation(
            messages
        )

        return updated_features


class WeightedGNN(nn.Module):
    def __init__(
        self,
        input_dim=3587,
        hidden_dim=256,
        num_layers=2,
        use_layer_norm=True,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError(
                "num_layers must be at least 1."
            )

        layers = []

        for layer_index in range(num_layers):
            layer_input_dim = (
                input_dim
                if layer_index == 0
                else hidden_dim
            )

            layers.append(
                WeightedGraphLayer(
                    input_dim=layer_input_dim,
                    output_dim=hidden_dim,
                    use_layer_norm=use_layer_norm,
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        node_features,
        normalized_edge_weights,
    ):
        if not torch.isfinite(
            node_features
        ).all():
            raise ValueError(
                "Node features contain NaN or Inf."
            )

        if not torch.isfinite(
            normalized_edge_weights
        ).all():
            raise ValueError(
                "Edge weights contain NaN or Inf."
            )

        edge_weights = (
            normalized_edge_weights.to(
                device=node_features.device,
                dtype=node_features.dtype,
            )
        )

        current_features = node_features
        layer_states = []

        for layer in self.layers:
            current_features = layer(
                node_features=current_features,
                normalized_edge_weights=(
                    edge_weights
                ),
            )
            layer_states.append(current_features)

        if not torch.isfinite(
            current_features
        ).all():
            raise RuntimeError(
                "GNN output contains NaN or Inf."
            )

        return WeightedGNNOutput(
            layer_states=tuple(layer_states),
            final_node_features=(
                current_features
            ),
        )