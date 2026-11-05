from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class GraphTokenOutput:
    graph_embedding: torch.Tensor
    graph_token: torch.Tensor


class MeanGraphReadout(nn.Module):
    def forward(self, node_features):
        if node_features.ndim not in (2, 3):
            raise ValueError(
                "node_features must have shape "
                "[K, D] or [B, K, D]."
            )

        if node_features.shape[-2] == 0:
            raise ValueError(
                "Cannot read out an empty graph."
            )

        return node_features.mean(
            dim=-2,
            keepdim=True,
        )


class GraphTokenProjector(nn.Module):
    def __init__(
        self,
        graph_dim=256,
        llm_hidden_dim=3584,
        use_layer_norm=True,
    ):
        super().__init__()

        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(
                graph_dim
            )
        else:
            self.layer_norm = nn.Identity()

        self.projection = nn.Linear(
            graph_dim,
            llm_hidden_dim,
            bias=False,
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(
            self.projection.weight
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

    def forward(self, graph_embedding):
        normalized = self.layer_norm(
            graph_embedding
        )
        return self.projection(normalized)


class GraphTokenHead(nn.Module):
    def __init__(
        self,
        graph_dim=256,
        llm_hidden_dim=3584,
        use_layer_norm=True,
    ):
        super().__init__()

        self.readout = MeanGraphReadout()
        self.projector = GraphTokenProjector(
            graph_dim=graph_dim,
            llm_hidden_dim=llm_hidden_dim,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, node_features):
        graph_embedding = self.readout(
            node_features
        )
        graph_token = self.projector(
            graph_embedding
        )

        if not torch.isfinite(
            graph_token
        ).all():
            raise RuntimeError(
                "Graph token contains NaN or Inf."
            )

        return GraphTokenOutput(
            graph_embedding=graph_embedding,
            graph_token=graph_token,
        )