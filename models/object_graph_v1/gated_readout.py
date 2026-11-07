import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class GatedGraphReadoutOutput:
    graph_embedding: torch.Tensor
    normalized_node_features: torch.Tensor
    graph_context: torch.Tensor
    attention_logits: torch.Tensor
    node_gates: torch.Tensor
    node_weights: torch.Tensor
    weighted_node_feature: torch.Tensor
    residual_mean_feature: torch.Tensor
    readout_gate: torch.Tensor
    projected_feature: torch.Tensor
    normalized_projected_feature: torch.Tensor
    graph_token: torch.Tensor
    token_scale: torch.Tensor


class GatedGraphReadout(nn.Module):
    """
    Convert relation-enhanced graph nodes into one
    LLM-compatible Graph Token.

    Node weights combine:
      1. learned attention logits,
      2. learned node gates,
      3. the question-conditioned node prior.
    """

    def __init__(
        self,
        hidden_dimension=256,
        node_context_dimension=9,
        graph_context_dimension=128,
        context_hidden_dimension=64,
        score_hidden_dimension=128,
        gate_hidden_dimension=64,
        llm_hidden_dimension=3584,
        dropout=0.1,
        prior_strength=0.5,
        initial_readout_gate=-1.0,
        initial_token_scale=0.35,
        eps=1e-6,
    ):
        super().__init__()

        positive_dimensions = {
            "hidden_dimension": hidden_dimension,
            "node_context_dimension": (
                node_context_dimension
            ),
            "graph_context_dimension": (
                graph_context_dimension
            ),
            "context_hidden_dimension": (
                context_hidden_dimension
            ),
            "score_hidden_dimension": (
                score_hidden_dimension
            ),
            "gate_hidden_dimension": (
                gate_hidden_dimension
            ),
            "llm_hidden_dimension": (
                llm_hidden_dimension
            ),
        }

        for name, value in (
            positive_dimensions.items()
        ):
            if int(value) <= 0:
                raise ValueError(
                    f"{name} must be positive."
                )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must lie in [0, 1)."
            )

        if prior_strength < 0.0:
            raise ValueError(
                "prior_strength must be "
                "non-negative."
            )

        if initial_token_scale <= 0.0:
            raise ValueError(
                "initial_token_scale must "
                "be positive."
            )

        if eps <= 0.0:
            raise ValueError(
                "eps must be positive."
            )

        self.hidden_dimension = int(
            hidden_dimension
        )
        self.node_context_dimension = int(
            node_context_dimension
        )
        self.graph_context_dimension = int(
            graph_context_dimension
        )
        self.llm_hidden_dimension = int(
            llm_hidden_dimension
        )
        self.prior_strength = float(
            prior_strength
        )
        self.eps = float(eps)

        self.node_norm = nn.LayerNorm(
            hidden_dimension
        )

        self.context_encoder = nn.Sequential(
            nn.Linear(
                node_context_dimension,
                context_hidden_dimension,
            ),
            nn.LayerNorm(
                context_hidden_dimension
            ),
            nn.GELU(),
        )

        self.graph_context_encoder = (
            nn.Sequential(
                nn.Linear(
                    hidden_dimension * 2,
                    graph_context_dimension,
                ),
                nn.GELU(),
                nn.LayerNorm(
                    graph_context_dimension
                ),
            )
        )

        self.attention_score = nn.Sequential(
            nn.Linear(
                hidden_dimension
                + graph_context_dimension
                + context_hidden_dimension,
                score_hidden_dimension,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                score_hidden_dimension,
                1,
            ),
        )

        self.node_gate = nn.Sequential(
            nn.Linear(
                hidden_dimension
                + context_hidden_dimension,
                gate_hidden_dimension,
            ),
            nn.GELU(),
            nn.Linear(
                gate_hidden_dimension,
                1,
            ),
        )

        readout_input_dimension = (
            hidden_dimension
            + graph_context_dimension
        )

        self.readout_candidate = nn.Sequential(
            nn.Linear(
                readout_input_dimension,
                hidden_dimension,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dimension,
                hidden_dimension,
            ),
        )

        self.readout_gate = nn.Sequential(
            nn.Linear(
                readout_input_dimension,
                gate_hidden_dimension,
            ),
            nn.GELU(),
            nn.Linear(
                gate_hidden_dimension,
                hidden_dimension,
            ),
        )

        self.graph_embedding_norm = (
            nn.LayerNorm(
                hidden_dimension
            )
        )

        self.token_projection = nn.Linear(
            hidden_dimension,
            llm_hidden_dimension,
        )

        self.log_token_scale = nn.Parameter(
            torch.tensor(
                math.log(
                    float(
                        initial_token_scale
                    )
                ),
                dtype=torch.float32,
            )
        )

        self._initialize_parameters(
            initial_readout_gate=(
                initial_readout_gate
            )
        )

    def _initialize_parameters(
        self,
        initial_readout_gate,
    ):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(
                    module.weight
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

        attention_output = (
            self.attention_score[-1]
        )

        nn.init.normal_(
            attention_output.weight,
            mean=0.0,
            std=0.01,
        )
        nn.init.zeros_(
            attention_output.bias
        )

        node_gate_output = (
            self.node_gate[-1]
        )

        nn.init.normal_(
            node_gate_output.weight,
            mean=0.0,
            std=0.01,
        )
        nn.init.zeros_(
            node_gate_output.bias
        )

        candidate_output = (
            self.readout_candidate[-1]
        )

        nn.init.xavier_uniform_(
            candidate_output.weight,
            gain=0.10,
        )
        nn.init.zeros_(
            candidate_output.bias
        )

        readout_gate_output = (
            self.readout_gate[-1]
        )

        nn.init.normal_(
            readout_gate_output.weight,
            mean=0.0,
            std=0.01,
        )
        nn.init.constant_(
            readout_gate_output.bias,
            float(initial_readout_gate),
        )

        nn.init.xavier_uniform_(
            self.token_projection.weight
        )
        nn.init.zeros_(
            self.token_projection.bias
        )

    def _validate_inputs(
        self,
        node_features,
        node_context,
        node_prior,
    ):
        if (
            node_features.ndim != 2
            or node_features.shape[1]
            != self.hidden_dimension
        ):
            raise ValueError(
                "node_features must have shape "
                f"(N, {self.hidden_dimension})."
            )

        if (
            node_context.ndim != 2
            or node_context.shape
            != (
                node_features.shape[0],
                self.node_context_dimension,
            )
        ):
            raise ValueError(
                "node_context must have shape "
                f"(N, {self.node_context_dimension})."
            )

        if (
            node_prior.ndim != 1
            or node_prior.shape[0]
            != node_features.shape[0]
        ):
            raise ValueError(
                "node_prior must have shape (N,)."
            )

        if node_features.shape[0] <= 0:
            raise ValueError(
                "At least one node is required."
            )

        tensors = (
            node_features,
            node_context,
            node_prior,
        )

        if not all(
            bool(torch.isfinite(tensor).all())
            for tensor in tensors
        ):
            raise ValueError(
                "A Readout input contains "
                "NaN or Inf."
            )

        if bool((node_prior < 0.0).any()):
            raise ValueError(
                "node_prior must be "
                "non-negative."
            )

    def forward(
        self,
        node_features,
        node_context,
        node_prior=None,
    ):
        device = node_features.device

        node_features = (
            node_features.float()
        )
        node_context = node_context.to(
            device=device,
            dtype=torch.float32,
        )

        if node_prior is None:
            node_prior = torch.ones(
                node_features.shape[0],
                device=device,
                dtype=torch.float32,
            )
        else:
            node_prior = node_prior.to(
                device=device,
                dtype=torch.float32,
            )

        self._validate_inputs(
            node_features=node_features,
            node_context=node_context,
            node_prior=node_prior,
        )

        normalized_nodes = self.node_norm(
            node_features
        )

        encoded_context = (
            self.context_encoder(
                node_context
            )
        )

        mean_feature = (
            normalized_nodes
            .mean(
                dim=0,
                keepdim=True,
            )
        )

        max_feature = (
            normalized_nodes
            .max(
                dim=0,
                keepdim=True,
            )
            .values
        )

        graph_context = (
            self.graph_context_encoder(
                torch.cat(
                    [
                        mean_feature,
                        max_feature,
                    ],
                    dim=-1,
                )
            )
        )

        repeated_graph_context = (
            graph_context.expand(
                normalized_nodes.shape[0],
                -1,
            )
        )

        attention_input = torch.cat(
            [
                normalized_nodes,
                repeated_graph_context,
                encoded_context,
            ],
            dim=-1,
        )

        attention_logits = (
            self.attention_score(
                attention_input
            )
            .squeeze(-1)
        )

        gate_input = torch.cat(
            [
                normalized_nodes,
                encoded_context,
            ],
            dim=-1,
        )

        node_gates = torch.sigmoid(
            self.node_gate(
                gate_input
            )
        ).squeeze(-1)

        prior_bias = (
            self.prior_strength
            * torch.log(
                node_prior.clamp_min(
                    self.eps
                )
            )
        )

        combined_logits = (
            attention_logits
            + torch.log(
                node_gates.clamp_min(
                    self.eps
                )
            )
            + prior_bias
        )

        node_weights = torch.softmax(
            combined_logits,
            dim=0,
        )

        weighted_node_feature = torch.sum(
            normalized_nodes
            * node_weights.unsqueeze(-1),
            dim=0,
            keepdim=True,
        )

        readout_input = torch.cat(
            [
                weighted_node_feature,
                graph_context,
            ],
            dim=-1,
        )

        candidate_embedding = (
            self.readout_candidate(
                readout_input
            )
        )

        readout_gate = torch.sigmoid(
            self.readout_gate(
                readout_input
            )
        )

        graph_embedding = (
            mean_feature
            + readout_gate
            * (
                candidate_embedding
                - mean_feature
            )
        )

        graph_embedding = (
            self.graph_embedding_norm(
                graph_embedding
            )
        )

        projected_feature = (
            self.token_projection(
                graph_embedding
            )
        )

        projected_rms = torch.sqrt(
            projected_feature
            .float()
            .square()
            .mean(
                dim=-1,
                keepdim=True,
            )
            + self.eps
        )

        normalized_projected_feature = (
            projected_feature
            / projected_rms
        )

        token_scale = torch.exp(
            self.log_token_scale
        )

        graph_token = (
            normalized_projected_feature
            * token_scale
        ).unsqueeze(1)

        output_tensors = (
            graph_embedding,
            normalized_nodes,
            graph_context,
            attention_logits,
            node_gates,
            node_weights,
            weighted_node_feature,
            mean_feature,
            readout_gate,
            projected_feature,
            normalized_projected_feature,
            graph_token,
            token_scale,
        )

        if not all(
            bool(torch.isfinite(tensor).all())
            for tensor in output_tensors
        ):
            raise RuntimeError(
                "A Readout output contains "
                "NaN or Inf."
            )

        return GatedGraphReadoutOutput(
            graph_embedding=graph_embedding,
            normalized_node_features=(
                normalized_nodes
            ),
            graph_context=graph_context,
            attention_logits=attention_logits,
            node_gates=node_gates,
            node_weights=node_weights,
            weighted_node_feature=(
                weighted_node_feature
            ),
            residual_mean_feature=(
                mean_feature
            ),
            readout_gate=readout_gate,
            projected_feature=(
                projected_feature
            ),
            normalized_projected_feature=(
                normalized_projected_feature
            ),
            graph_token=graph_token,
            token_scale=token_scale,
        )