from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
from torch import nn

from vlm.detectors.detection_types import (
    DetectionBox,
)
from vlm.models.object_graph_v1.attention_region_proposer import (
    AttentionRegionProposalOutput,
    AttentionRegionProposer,
)
from vlm.models.object_graph_v1.gated_readout import (
    GatedGraphReadout,
    GatedGraphReadoutOutput,
)
from vlm.models.object_graph_v1.hybrid_node_builder import (
    HybridNodeBuilder,
    HybridNodeBuilderOutput,
)
from vlm.models.object_graph_v1.region_feature_pooler import (
    RegionFeaturePooler,
    RegionFeaturePoolerOutput,
)
from vlm.models.object_graph_v1.residual_gated_gnn import (
    ResidualGatedGNN,
    ResidualGatedGNNOutput,
)
from vlm.models.object_graph_v1.sparse_graph_builder import (
    SparseObjectGraphBuilder,
    SparseObjectGraphOutput,
)


@dataclass
class ObjectGraphTokenEncoderV1Output:
    graph_token: torch.Tensor
    proposal_output: (
        AttentionRegionProposalOutput
    )
    detection_pool_output: Optional[
        RegionFeaturePoolerOutput
    ]
    attention_pool_output: Optional[
        RegionFeaturePoolerOutput
    ]
    hybrid_output: HybridNodeBuilderOutput
    graph_output: SparseObjectGraphOutput
    class_indices: torch.Tensor
    class_embeddings: torch.Tensor
    gnn_input_features: torch.Tensor
    gnn_output: ResidualGatedGNNOutput
    readout_output: GatedGraphReadoutOutput


class ObjectGraphTokenEncoderV1(nn.Module):
    """
    Object–Attention Hybrid Graph Token Encoder.

    Pipeline:
      YOLO detections + question-conditioned Attention
      -> compact Attention Regions
      -> region feature pooling
      -> Hybrid Graph Nodes
      -> sparse relation graph
      -> residual gated GNN
      -> gated readout
      -> one LLM Graph Token
    """

    def __init__(
        self,
        visual_dimension=3584,
        llm_hidden_dimension=3584,
        class_count=80,
        class_embedding_dimension=32,
        token_grid_size=16,
        hidden_dimension=256,
        gnn_layer_count=2,
        max_detection_nodes=10,
        max_attention_nodes=2,
        max_total_nodes=12,
        max_nodes_per_class=4,
        graph_top_k=3,
        dropout=0.1,
        initial_token_scale=0.35,
    ):
        super().__init__()

        positive_values = {
            "visual_dimension": (
                visual_dimension
            ),
            "llm_hidden_dimension": (
                llm_hidden_dimension
            ),
            "class_count": class_count,
            "class_embedding_dimension": (
                class_embedding_dimension
            ),
            "token_grid_size": (
                token_grid_size
            ),
            "hidden_dimension": (
                hidden_dimension
            ),
            "gnn_layer_count": (
                gnn_layer_count
            ),
            "max_detection_nodes": (
                max_detection_nodes
            ),
            "max_attention_nodes": (
                max_attention_nodes
            ),
            "max_total_nodes": (
                max_total_nodes
            ),
            "max_nodes_per_class": (
                max_nodes_per_class
            ),
            "graph_top_k": graph_top_k,
        }

        for name, value in (
            positive_values.items()
        ):
            if int(value) <= 0:
                raise ValueError(
                    f"{name} must be positive."
                )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must lie in [0, 1)."
            )

        if initial_token_scale <= 0.0:
            raise ValueError(
                "initial_token_scale must "
                "be positive."
            )

        self.visual_dimension = int(
            visual_dimension
        )
        self.llm_hidden_dimension = int(
            llm_hidden_dimension
        )
        self.class_count = int(
            class_count
        )
        self.class_embedding_dimension = int(
            class_embedding_dimension
        )
        self.token_grid_size = int(
            token_grid_size
        )
        self.hidden_dimension = int(
            hidden_dimension
        )

        self.geometry_dimension = 6
        self.scalar_dimension = 9

        self.raw_node_dimension = (
            self.visual_dimension
            + self.geometry_dimension
            + self.scalar_dimension
        )

        self.gnn_input_dimension = (
            self.raw_node_dimension
            + self.class_embedding_dimension
        )

        # class_count is reserved as the unknown
        # index for Attention-only nodes.
        self.unknown_class_index = (
            self.class_count
        )

        self.attention_region_proposer = (
            AttentionRegionProposer(
                token_grid_size=(
                    token_grid_size
                ),
                peak_relative_threshold=0.15,
                peak_nms_kernel_size=3,
                growth_relative_threshold=0.55,
                max_region_radius=2,
                smoothing_kernel_size=3,
                min_region_patches=1,
                min_region_mass=0.005,
                max_regions=4,
                border_margin=1,
                border_penalty=0.50,
            )
        )

        self.region_feature_pooler = (
            RegionFeaturePooler(
                token_grid_size=(
                    token_grid_size
                ),
                eps=1e-12,
            )
        )

        self.hybrid_node_builder = (
            HybridNodeBuilder(
                min_detection_confidence=0.20,
                attention_rescue_mass=0.01,
                attention_rescue_peak=0.01,
                min_attention_region_mass=0.01,
                attention_residual_relative_score=(
                    0.15
                ),
                match_overlap_threshold=0.30,
                duplicate_iou_threshold=0.70,
                max_detection_nodes=(
                    max_detection_nodes
                ),
                max_attention_nodes=(
                    max_attention_nodes
                ),
                max_total_nodes=(
                    max_total_nodes
                ),
                max_nodes_per_class=(
                    max_nodes_per_class
                ),
                attention_support_scale=0.03,
                proposal_score_scale=0.01,
                max_attention_visual_blend=(
                    0.35
                ),
                max_region_visual_blend=0.20,
                attention_only_visual_blend=(
                    0.75
                ),
                eps=1e-12,
            )
        )

        self.sparse_graph_builder = (
            SparseObjectGraphBuilder(
                top_k=graph_top_k,
                minimum_neighbors=1,
                minimum_edge_score=0.15,
                spatial_sigma=0.35,
                self_loop_weight=1.0,
                spatial_weight=0.25,
                visual_weight=0.20,
                attention_weight=0.20,
                overlap_weight=0.10,
                size_weight=0.05,
                shared_region_weight=0.10,
                relevance_weight=0.10,
                eps=1e-12,
            )
        )

        self.class_embedding = nn.Embedding(
            self.class_count + 1,
            self.class_embedding_dimension,
        )

        self.gnn = ResidualGatedGNN(
            input_dimension=(
                self.gnn_input_dimension
            ),
            relation_dimension=17,
            hidden_dimension=(
                hidden_dimension
            ),
            layer_count=gnn_layer_count,
            relation_hidden_dimension=64,
            gate_hidden_dimension=64,
            dropout=dropout,
            initial_residual_gate=-2.0,
            eps=1e-12,
        )

        self.readout = GatedGraphReadout(
            hidden_dimension=hidden_dimension,
            node_context_dimension=(
                self.scalar_dimension
            ),
            graph_context_dimension=128,
            context_hidden_dimension=64,
            score_hidden_dimension=128,
            gate_hidden_dimension=64,
            llm_hidden_dimension=(
                llm_hidden_dimension
            ),
            dropout=dropout,
            prior_strength=0.5,
            initial_readout_gate=-1.0,
            initial_token_scale=(
                initial_token_scale
            ),
            eps=1e-6,
        )

        nn.init.normal_(
            self.class_embedding.weight,
            mean=0.0,
            std=0.02,
        )

    def _class_indices(
        self,
        nodes,
        device,
    ):
        indices = []

        for node in nodes:
            class_id = int(node.class_id)

            if class_id == -1:
                class_index = (
                    self.unknown_class_index
                )
            elif (
                0 <= class_id
                < self.class_count
            ):
                class_index = class_id
            else:
                raise ValueError(
                    "Class ID is outside the "
                    "configured class range: "
                    f"{class_id}."
                )

            indices.append(class_index)

        return torch.tensor(
            indices,
            device=device,
            dtype=torch.long,
        )

    def forward(
        self,
        detections: Sequence[DetectionBox],
        local_attention: torch.Tensor,
        projected_visual_features:
            torch.Tensor,
        attention_profiles: torch.Tensor,
        tile_grid: Tuple[int, int],
        image_size: Tuple[int, int],
    ) -> ObjectGraphTokenEncoderV1Output:
        detections = tuple(detections)

        proposal_output = (
            self.attention_region_proposer(
                local_attention=(
                    local_attention
                ),
                tile_grid=tile_grid,
                image_size=image_size,
            )
        )

        attention_regions = (
            proposal_output.regions
        )

        detection_pool_output = None
        detection_features = tuple()

        if detections:
            detection_pool_output = (
                self.region_feature_pooler(
                    normalized_boxes=[
                        detection
                        .normalized_xyxy
                        for detection
                        in detections
                    ],
                    local_attention=(
                        local_attention
                    ),
                    projected_visual_features=(
                        projected_visual_features
                    ),
                    attention_profiles=(
                        attention_profiles
                    ),
                    tile_grid=tile_grid,
                )
            )

            detection_features = (
                detection_pool_output
                .region_features
            )

        attention_pool_output = None
        attention_region_features = tuple()

        if attention_regions:
            attention_pool_output = (
                self.region_feature_pooler(
                    normalized_boxes=[
                        region.normalized_xyxy
                        for region
                        in attention_regions
                    ],
                    local_attention=(
                        local_attention
                    ),
                    projected_visual_features=(
                        projected_visual_features
                    ),
                    attention_profiles=(
                        attention_profiles
                    ),
                    tile_grid=tile_grid,
                )
            )

            attention_region_features = (
                attention_pool_output
                .region_features
            )

        hybrid_output = (
            self.hybrid_node_builder(
                detections=detections,
                detection_features=(
                    detection_features
                ),
                attention_regions=(
                    attention_regions
                ),
                attention_region_features=(
                    attention_region_features
                ),
            )
        )

        graph_output = (
            self.sparse_graph_builder(
                hybrid_output.nodes
            )
        )

        encoder_device = (
            self.class_embedding
            .weight
            .device
        )

        class_indices = (
            self._class_indices(
                nodes=hybrid_output.nodes,
                device=encoder_device,
            )
        )

        class_embeddings = (
            self.class_embedding(
                class_indices
            )
        )

        raw_node_features = (
            graph_output.node_features.to(
                device=encoder_device,
                dtype=torch.float32,
            )
        )

        if raw_node_features.shape[-1] != (
            self.raw_node_dimension
        ):
            raise ValueError(
                "Unexpected raw node dimension: "
                f"{raw_node_features.shape[-1]}; "
                f"expected "
                f"{self.raw_node_dimension}."
            )

        gnn_input_features = torch.cat(
            [
                raw_node_features,
                class_embeddings,
            ],
            dim=-1,
        )

        gnn_output = self.gnn(
            node_features=gnn_input_features,
            edge_index=(
                graph_output.edge_index
            ),
            edge_weight=(
                graph_output.edge_weight
            ),
            edge_features=(
                graph_output.edge_features
            ),
        )

        node_context = torch.stack(
            [
                node.scalar_feature
                for node
                in hybrid_output.nodes
            ],
            dim=0,
        ).to(
            device=encoder_device,
            dtype=torch.float32,
        )

        node_prior = torch.tensor(
            [
                node.selection_score
                for node
                in hybrid_output.nodes
            ],
            device=encoder_device,
            dtype=torch.float32,
        )

        readout_output = self.readout(
            node_features=(
                gnn_output
                .final_node_features
            ),
            node_context=node_context,
            node_prior=node_prior,
        )

        graph_token = (
            readout_output.graph_token
        )

        expected_shape = (
            1,
            1,
            self.llm_hidden_dimension,
        )

        if tuple(graph_token.shape) != (
            expected_shape
        ):
            raise RuntimeError(
                "Unexpected Graph Token shape: "
                f"{tuple(graph_token.shape)}; "
                f"expected {expected_shape}."
            )

        if not bool(
            torch.isfinite(graph_token).all()
        ):
            raise RuntimeError(
                "Graph Token contains NaN or Inf."
            )

        return ObjectGraphTokenEncoderV1Output(
            graph_token=graph_token,
            proposal_output=proposal_output,
            detection_pool_output=(
                detection_pool_output
            ),
            attention_pool_output=(
                attention_pool_output
            ),
            hybrid_output=hybrid_output,
            graph_output=graph_output,
            class_indices=class_indices,
            class_embeddings=(
                class_embeddings
            ),
            gnn_input_features=(
                gnn_input_features
            ),
            gnn_output=gnn_output,
            readout_output=readout_output,
        )