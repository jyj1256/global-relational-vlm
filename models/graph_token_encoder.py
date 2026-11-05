from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from vlm.models.graph_node_builder import (
    GraphNodeBatch,
    build_graph_nodes,
)
from vlm.models.graph_readout import (
    GraphTokenHead,
    GraphTokenOutput,
)
from vlm.models.salient_region_selector import (
    SalientRegionSelection,
    select_salient_regions,
)
from vlm.models.weighted_gnn import (
    WeightedGNN,
    WeightedGNNOutput,
)
from vlm.models.weighted_graph_builder import (
    WeightedGraph,
    build_weighted_graph,
)


@dataclass
class GraphTokenEncoderOutput:
    graph_token: torch.Tensor
    selection: SalientRegionSelection
    graph_nodes: GraphNodeBatch
    selected_attention_profiles: torch.Tensor
    weighted_graph: WeightedGraph
    gnn_output: WeightedGNNOutput
    token_output: GraphTokenOutput


class GraphTokenEncoder(nn.Module):
    """
    Complete Graph Token Encoder.

    Inputs:
        local_attention:
            (number_of_local_tokens,)

        attention_profiles:
            (number_of_local_tokens, 8, 28)
            or
            (number_of_local_tokens, 224)

        projected_visual_features:
            (number_of_image_inputs, 256, 3584)

    Output:
        One Graph Token with shape (1, 1, 3584)
    """

    def __init__(
        self,
        top_k: int = 16,
        token_grid=(16, 16),
        sigma_p: float = 0.25,
        threshold: float = 0.70,
        spatial_weight: float = 0.35,
        visual_weight: float = 0.20,
        attention_weight: float = 0.45,
        node_input_dimension: int = 3587,
        graph_dimension: int = 256,
        gnn_layers: int = 2,
        llm_hidden_dimension: int = 3584,
    ):
        super().__init__()

        self.top_k = top_k
        self.token_grid = tuple(token_grid)

        self.sigma_p = sigma_p
        self.threshold = threshold
        self.spatial_weight = spatial_weight
        self.visual_weight = visual_weight
        self.attention_weight = attention_weight

        self.node_input_dimension = (
            node_input_dimension
        )
        self.graph_dimension = graph_dimension
        self.llm_hidden_dimension = (
            llm_hidden_dimension
        )

        self.gnn = WeightedGNN(
            input_dim=node_input_dimension,
            hidden_dim=graph_dimension,
            num_layers=gnn_layers,
            use_layer_norm=True,
        )

        self.graph_head = GraphTokenHead(
            graph_dim=graph_dimension,
            llm_hidden_dim=llm_hidden_dimension,
            use_layer_norm=True,
        )

    def _prepare_local_attention(
        self,
        local_attention,
    ):
        if torch.is_tensor(local_attention):
            local_attention = (
                local_attention
                .detach()
                .float()
                .cpu()
                .numpy()
            )
        else:
            local_attention = np.asarray(
                local_attention,
                dtype=np.float32,
            )

        local_attention = local_attention.reshape(-1)

        if not np.isfinite(local_attention).all():
            raise ValueError(
                "local_attention contains NaN or Inf."
            )

        if local_attention.size == 0:
            raise ValueError(
                "local_attention is empty."
            )

        return local_attention

    def _prepare_attention_profiles(
        self,
        attention_profiles,
        expected_token_count,
    ):
        attention_profiles = torch.as_tensor(
            attention_profiles
        ).float()

        if attention_profiles.ndim == 3:
            attention_profiles = (
                attention_profiles.flatten(
                    start_dim=1
                )
            )

        elif attention_profiles.ndim != 2:
            raise ValueError(
                "attention_profiles must have shape "
                "(N, L, H) or (N, L*H)."
            )

        if attention_profiles.shape[0] != (
            expected_token_count
        ):
            raise ValueError(
                "Attention profile token count does "
                "not match local attention length: "
                f"{attention_profiles.shape[0]} vs "
                f"{expected_token_count}."
            )

        return attention_profiles

    def forward(
        self,
        local_attention,
        attention_profiles,
        projected_visual_features,
        tile_grid,
        image_size,
    ) -> GraphTokenEncoderOutput:
        tile_grid = tuple(
            int(value) for value in tile_grid
        )

        image_size = tuple(
            int(value) for value in image_size
        )

        if len(tile_grid) != 2:
            raise ValueError(
                "tile_grid must contain two values."
            )

        if len(image_size) != 2:
            raise ValueError(
                "image_size must contain width and height."
            )

        num_local_tiles = (
            tile_grid[0] * tile_grid[1]
        )

        local_attention = (
            self._prepare_local_attention(
                local_attention
            )
        )

        attention_profiles = (
            self._prepare_attention_profiles(
                attention_profiles,
                expected_token_count=(
                    local_attention.shape[0]
                ),
            )
        )

        projected_visual_features = (
            torch.as_tensor(
                projected_visual_features
            )
        )

        if projected_visual_features.ndim != 3:
            raise ValueError(
                "projected_visual_features must have "
                "shape (T, 256, 3584)."
            )

        if projected_visual_features.shape[0] < (
            num_local_tiles
        ):
            raise ValueError(
                "Not enough visual feature tiles."
            )

        selection = select_salient_regions(
            local_attention=local_attention,
            tile_grid=tile_grid,
            image_size=image_size,
            token_grid=self.token_grid,
            top_k=self.top_k,
        )

        graph_nodes = build_graph_nodes(
            projected_visual_features=(
                projected_visual_features
            ),
            selection=selection,
            num_local_tiles=num_local_tiles,
        )

        selected_indices = (
            graph_nodes.local_token_indices
            .long()
            .to(attention_profiles.device)
        )

        selected_attention_profiles = (
            attention_profiles.index_select(
                dim=0,
                index=selected_indices,
            )
        )

        graph_device = next(
            self.gnn.parameters()
        ).device

        graph_dtype = next(
            self.gnn.parameters()
        ).dtype

        visual_features = (
            graph_nodes.visual_features.to(
                device=graph_device,
                dtype=graph_dtype,
            )
        )

        normalized_positions = (
            graph_nodes.normalized_positions.to(
                device=graph_device,
                dtype=graph_dtype,
            )
        )

        attention_scores = (
            graph_nodes.attention_scores.to(
                device=graph_device,
                dtype=graph_dtype,
            )
        )

        selected_attention_profiles = (
            selected_attention_profiles.to(
                device=graph_device,
                dtype=graph_dtype,
            )
        )

        weighted_graph = build_weighted_graph(
            visual_features=visual_features,
            normalized_positions=(
                normalized_positions
            ),
            attention_scores=attention_scores,
            attention_profiles=(
                selected_attention_profiles
            ),
            sigma_p=self.sigma_p,
            threshold=self.threshold,
            spatial_weight=self.spatial_weight,
            visual_weight=self.visual_weight,
            attention_weight=(
                self.attention_weight
            ),
        )

        if weighted_graph.node_features.shape[-1] != (
            self.node_input_dimension
        ):
            raise RuntimeError(
                "Unexpected graph node dimension: "
                f"{weighted_graph.node_features.shape[-1]}"
            )

        gnn_output = self.gnn(
            node_features=(
                weighted_graph.node_features
            ),
            normalized_edge_weights=(
                weighted_graph
                .normalized_edge_weights
            ),
        )

        token_output = self.graph_head(
            gnn_output.final_node_features
        )

        graph_token = (
            token_output.graph_token.unsqueeze(1)
        )

        return GraphTokenEncoderOutput(
            graph_token=graph_token,
            selection=selection,
            graph_nodes=graph_nodes,
            selected_attention_profiles=(
                selected_attention_profiles
            ),
            weighted_graph=weighted_graph,
            gnn_output=gnn_output,
            token_output=token_output,
        )