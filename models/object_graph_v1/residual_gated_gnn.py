from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


@dataclass
class ResidualGatedLayerOutput:
    node_features: torch.Tensor
    aggregated_messages: torch.Tensor
    edge_messages: torch.Tensor
    message_gates: torch.Tensor
    residual_gates: torch.Tensor


@dataclass
class ResidualGatedGNNOutput:
    input_node_features: torch.Tensor
    projected_node_features: torch.Tensor
    layer_outputs: Tuple[
        ResidualGatedLayerOutput,
        ...,
    ]
    final_node_features: torch.Tensor


class ResidualGatedGraphLayer(nn.Module):
    """
    One relation-aware graph layer.

    Edge gates control individual messages, while
    residual gates limit how much aggregated graph
    information changes each node.
    """

    def __init__(
        self,
        hidden_dimension=256,
        relation_dimension=17,
        relation_hidden_dimension=64,
        gate_hidden_dimension=64,
        dropout=0.1,
        initial_residual_gate=-2.0,
        eps=1e-12,
    ):
        super().__init__()

        if hidden_dimension <= 0:
            raise ValueError(
                "hidden_dimension must be positive."
            )

        if relation_dimension <= 0:
            raise ValueError(
                "relation_dimension must be positive."
            )

        if relation_hidden_dimension <= 0:
            raise ValueError(
                "relation_hidden_dimension must "
                "be positive."
            )

        if gate_hidden_dimension <= 0:
            raise ValueError(
                "gate_hidden_dimension must "
                "be positive."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must lie in [0, 1)."
            )

        if eps <= 0.0:
            raise ValueError(
                "eps must be positive."
            )

        self.hidden_dimension = int(
            hidden_dimension
        )
        self.relation_dimension = int(
            relation_dimension
        )
        self.relation_hidden_dimension = int(
            relation_hidden_dimension
        )
        self.eps = float(eps)

        self.relation_encoder = nn.Sequential(
            nn.Linear(
                relation_dimension,
                relation_hidden_dimension,
            ),
            nn.LayerNorm(
                relation_hidden_dimension
            ),
            nn.GELU(),
        )

        self.message_network = nn.Sequential(
            nn.Linear(
                hidden_dimension
                + relation_hidden_dimension,
                hidden_dimension,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dimension,
                hidden_dimension,
            ),
        )

        self.message_gate = nn.Sequential(
            nn.Linear(
                hidden_dimension * 2
                + relation_hidden_dimension
                + 1,
                gate_hidden_dimension,
            ),
            nn.GELU(),
            nn.Linear(
                gate_hidden_dimension,
                1,
            ),
        )

        self.update_network = nn.Sequential(
            nn.Linear(
                hidden_dimension * 2,
                hidden_dimension,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dimension,
                hidden_dimension,
            ),
        )

        self.residual_gate = nn.Sequential(
            nn.Linear(
                hidden_dimension * 2,
                gate_hidden_dimension,
            ),
            nn.GELU(),
            nn.Linear(
                gate_hidden_dimension,
                hidden_dimension,
            ),
        )

        self.output_dropout = nn.Dropout(
            dropout
        )
        self.output_norm = nn.LayerNorm(
            hidden_dimension
        )

        self._initialize_parameters(
            initial_residual_gate
        )

    def _initialize_parameters(
        self,
        initial_residual_gate,
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

        message_gate_output = (
            self.message_gate[-1]
        )

        nn.init.zeros_(
            message_gate_output.bias
        )

        update_output = (
            self.update_network[-1]
        )

        nn.init.xavier_uniform_(
            update_output.weight,
            gain=0.10,
        )
        nn.init.zeros_(
            update_output.bias
        )

        residual_gate_output = (
            self.residual_gate[-1]
        )

        nn.init.normal_(
            residual_gate_output.weight,
            mean=0.0,
            std=0.01,
        )
        nn.init.constant_(
            residual_gate_output.bias,
            float(initial_residual_gate),
        )

    def _validate_inputs(
        self,
        node_features,
        edge_index,
        edge_weight,
        edge_features,
    ):
        if node_features.ndim != 2:
            raise ValueError(
                "node_features must have "
                "shape (N, H)."
            )

        if node_features.shape[1] != (
            self.hidden_dimension
        ):
            raise ValueError(
                "Unexpected hidden dimension: "
                f"{node_features.shape[1]}."
            )

        if (
            edge_index.ndim != 2
            or edge_index.shape[0] != 2
        ):
            raise ValueError(
                "edge_index must have "
                "shape (2, E)."
            )

        edge_count = edge_index.shape[1]

        if (
            edge_weight.ndim != 1
            or edge_weight.shape[0]
            != edge_count
        ):
            raise ValueError(
                "edge_weight must have "
                "shape (E,)."
            )

        if (
            edge_features.ndim != 2
            or edge_features.shape
            != (
                edge_count,
                self.relation_dimension,
            )
        ):
            raise ValueError(
                "edge_features must have "
                "shape (E, R)."
            )

        if edge_count > 0:
            if int(edge_index.min().item()) < 0:
                raise ValueError(
                    "edge_index contains a "
                    "negative node index."
                )

            if int(edge_index.max().item()) >= (
                node_features.shape[0]
            ):
                raise ValueError(
                    "edge_index exceeds the "
                    "node count."
                )

        tensors = (
            node_features,
            edge_weight,
            edge_features,
        )

        if not all(
            bool(torch.isfinite(tensor).all())
            for tensor in tensors
        ):
            raise ValueError(
                "A GNN input contains NaN or Inf."
            )

        if bool((edge_weight < 0.0).any()):
            raise ValueError(
                "edge_weight must be "
                "non-negative."
            )

    def forward(
        self,
        node_features,
        edge_index,
        edge_weight,
        edge_features,
    ):
        self._validate_inputs(
            node_features=node_features,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_features=edge_features,
        )

        node_count = node_features.shape[0]
        edge_count = edge_index.shape[1]

        if edge_count == 0:
            empty_messages = torch.empty(
                0,
                self.hidden_dimension,
                device=node_features.device,
                dtype=node_features.dtype,
            )
            empty_gates = torch.empty(
                0,
                device=node_features.device,
                dtype=node_features.dtype,
            )
            zero_aggregate = torch.zeros_like(
                node_features
            )
            zero_residual_gates = (
                torch.zeros_like(
                    node_features
                )
            )

            return ResidualGatedLayerOutput(
                node_features=node_features,
                aggregated_messages=(
                    zero_aggregate
                ),
                edge_messages=empty_messages,
                message_gates=empty_gates,
                residual_gates=(
                    zero_residual_gates
                ),
            )

        source_indices = edge_index[0]
        target_indices = edge_index[1]

        source_features = (
            node_features.index_select(
                dim=0,
                index=source_indices,
            )
        )
        target_features = (
            node_features.index_select(
                dim=0,
                index=target_indices,
            )
        )

        encoded_relations = (
            self.relation_encoder(
                edge_features
            )
        )

        message_input = torch.cat(
            [
                source_features,
                encoded_relations,
            ],
            dim=-1,
        )

        raw_messages = (
            self.message_network(
                message_input
            )
        )

        gate_input = torch.cat(
            [
                source_features,
                target_features,
                encoded_relations,
                edge_weight.unsqueeze(-1),
            ],
            dim=-1,
        )

        message_gates = torch.sigmoid(
            self.message_gate(
                gate_input
            )
        ).squeeze(-1)

        effective_weights = (
            edge_weight
            * message_gates
        )

        edge_messages = (
            raw_messages
            * effective_weights.unsqueeze(-1)
        )

        # Aggregate in a fixed node and edge order.
        #
        # CUDA index_add_ uses atomic additions and
        # can vary by a few floating-point ULPs across
        # otherwise identical runs. Graphs here contain
        # at most 12 nodes, so deterministic per-node
        # reduction has negligible computational cost.
        message_rows = []
        normalization_rows = []

        normalization_weights = (
            edge_weight.to(
                node_features.dtype
            )
        )

        for node_index in range(node_count):
            node_edge_mask = (
                target_indices
                == node_index
            )

            if bool(
                node_edge_mask
                .any()
                .item()
            ):
                message_rows.append(
                    edge_messages[
                        node_edge_mask
                    ].sum(dim=0)
                )

                normalization_rows.append(
                    normalization_weights[
                        node_edge_mask
                    ].sum()
                )
            else:
                message_rows.append(
                    torch.zeros(
                        self.hidden_dimension,
                        device=(
                            node_features.device
                        ),
                        dtype=(
                            node_features.dtype
                        ),
                    )
                )

                normalization_rows.append(
                    torch.zeros(
                        (),
                        device=(
                            node_features.device
                        ),
                        dtype=(
                            node_features.dtype
                        ),
                    )
                )

        aggregated_messages = torch.stack(
            message_rows,
            dim=0,
        )

        # Normalize using the original edge weights.
        # A low learned gate therefore still reduces
        # the total message magnitude.
        normalization = torch.stack(
            normalization_rows,
            dim=0,
        )

        aggregated_messages = (
            aggregated_messages
            / normalization
            .clamp_min(self.eps)
            .unsqueeze(-1)
        )

        update_input = torch.cat(
            [
                node_features,
                aggregated_messages,
            ],
            dim=-1,
        )

        candidate_update = (
            self.update_network(
                update_input
            )
        )

        residual_gates = torch.sigmoid(
            self.residual_gate(
                update_input
            )
        )

        gated_update = (
            residual_gates
            * candidate_update
        )

        output_features = self.output_norm(
            node_features
            + self.output_dropout(
                gated_update
            )
        )

        tensors = (
            output_features,
            aggregated_messages,
            edge_messages,
            message_gates,
            residual_gates,
        )

        if not all(
            bool(torch.isfinite(tensor).all())
            for tensor in tensors
        ):
            raise RuntimeError(
                "A GNN layer output contains "
                "NaN or Inf."
            )

        return ResidualGatedLayerOutput(
            node_features=output_features,
            aggregated_messages=(
                aggregated_messages
            ),
            edge_messages=edge_messages,
            message_gates=message_gates,
            residual_gates=residual_gates,
        )


class ResidualGatedGNN(nn.Module):
    """
    Project raw Hybrid Nodes and process them with
    multiple residual gated graph layers.
    """

    def __init__(
        self,
        input_dimension,
        relation_dimension=17,
        hidden_dimension=256,
        layer_count=2,
        relation_hidden_dimension=64,
        gate_hidden_dimension=64,
        dropout=0.1,
        initial_residual_gate=-2.0,
        eps=1e-12,
    ):
        super().__init__()

        if input_dimension <= 0:
            raise ValueError(
                "input_dimension must be positive."
            )

        if layer_count <= 0:
            raise ValueError(
                "layer_count must be positive."
            )

        self.input_dimension = int(
            input_dimension
        )
        self.relation_dimension = int(
            relation_dimension
        )
        self.hidden_dimension = int(
            hidden_dimension
        )
        self.layer_count = int(
            layer_count
        )

        self.input_projection = nn.Sequential(
            nn.Linear(
                input_dimension,
                hidden_dimension,
            ),
            nn.GELU(),
            nn.LayerNorm(
                hidden_dimension
            ),
            nn.Dropout(dropout),
        )

        self.layers = nn.ModuleList(
            [
                ResidualGatedGraphLayer(
                    hidden_dimension=(
                        hidden_dimension
                    ),
                    relation_dimension=(
                        relation_dimension
                    ),
                    relation_hidden_dimension=(
                        relation_hidden_dimension
                    ),
                    gate_hidden_dimension=(
                        gate_hidden_dimension
                    ),
                    dropout=dropout,
                    initial_residual_gate=(
                        initial_residual_gate
                    ),
                    eps=eps,
                )
                for _ in range(layer_count)
            ]
        )

        input_linear = (
            self.input_projection[0]
        )

        nn.init.xavier_uniform_(
            input_linear.weight
        )
        nn.init.zeros_(
            input_linear.bias
        )

    def forward(
        self,
        node_features,
        edge_index,
        edge_weight,
        edge_features,
    ):
        if node_features.ndim != 2:
            raise ValueError(
                "node_features must have "
                "shape (N, D)."
            )

        if node_features.shape[1] != (
            self.input_dimension
        ):
            raise ValueError(
                "Unexpected input dimension: "
                f"{node_features.shape[1]}; "
                f"expected {self.input_dimension}."
            )

        input_node_features = (
            node_features.float()
        )

        projected_node_features = (
            self.input_projection(
                input_node_features
            )
        )

        current_features = (
            projected_node_features
        )
        layer_outputs = []

        edge_weight = edge_weight.to(
            device=current_features.device,
            dtype=current_features.dtype,
        )
        edge_features = edge_features.to(
            device=current_features.device,
            dtype=current_features.dtype,
        )
        edge_index = edge_index.to(
            current_features.device
        )

        for layer in self.layers:
            layer_output = layer(
                node_features=current_features,
                edge_index=edge_index,
                edge_weight=edge_weight,
                edge_features=edge_features,
            )

            layer_outputs.append(
                layer_output
            )
            current_features = (
                layer_output.node_features
            )

        return ResidualGatedGNNOutput(
            input_node_features=(
                input_node_features
            ),
            projected_node_features=(
                projected_node_features
            ),
            layer_outputs=tuple(
                layer_outputs
            ),
            final_node_features=(
                current_features
            ),
        )