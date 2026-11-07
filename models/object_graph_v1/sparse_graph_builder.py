from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from vlm.models.object_graph_v1.hybrid_node_builder import (
    HybridGraphNode,
)


@dataclass
class SparseGraphEdge:
    edge_id: int
    first_node: int
    second_node: int
    weight: float
    forced: bool
    spatial_similarity: float
    visual_similarity: float
    attention_similarity: float
    overlap_iou: float
    size_similarity: float
    shared_attention_region: float
    relevance_similarity: float


@dataclass
class SparseObjectGraphOutput:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    edge_features: torch.Tensor
    dense_edge_scores: torch.Tensor
    sparse_adjacency: torch.Tensor
    normalized_adjacency: torch.Tensor
    edge_mask: torch.Tensor
    degree: torch.Tensor
    spatial_similarity: torch.Tensor
    visual_similarity: torch.Tensor
    attention_similarity: torch.Tensor
    overlap_iou: torch.Tensor
    size_similarity: torch.Tensor
    shared_attention_region: torch.Tensor
    relevance_similarity: torch.Tensor
    edges: Tuple[SparseGraphEdge, ...]
    relation_dimension: int
    undirected_edge_count: int
    directed_edge_count: int


def _box_iou(first_box, second_box):
    ax1, ay1, ax2, ay2 = first_box
    bx1, by1, bx2, by2 = second_box

    intersection_width = max(
        0.0,
        min(ax2, bx2) - max(ax1, bx1),
    )
    intersection_height = max(
        0.0,
        min(ay2, by2) - max(ay1, by1),
    )

    intersection = (
        intersection_width
        * intersection_height
    )

    first_area = max(
        0.0,
        ax2 - ax1,
    ) * max(
        0.0,
        ay2 - ay1,
    )
    second_area = max(
        0.0,
        bx2 - bx1,
    ) * max(
        0.0,
        by2 - by1,
    )

    union = (
        first_area
        + second_area
        - intersection
    )

    if union <= 0.0:
        return 0.0

    return intersection / union


class SparseObjectGraphBuilder(nn.Module):
    """
    Build a sparse, undirected object graph.

    Topology is selected using fixed interpretable
    similarities. Directed relation features are
    generated for the later GNN.
    """

    def __init__(
        self,
        top_k=3,
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
    ):
        super().__init__()

        if top_k <= 0:
            raise ValueError(
                "top_k must be positive."
            )

        if minimum_neighbors <= 0:
            raise ValueError(
                "minimum_neighbors must be positive."
            )

        if minimum_neighbors > top_k:
            raise ValueError(
                "minimum_neighbors cannot exceed "
                "top_k."
            )

        if not 0.0 <= minimum_edge_score <= 1.0:
            raise ValueError(
                "minimum_edge_score must lie "
                "in [0, 1]."
            )

        if spatial_sigma <= 0.0:
            raise ValueError(
                "spatial_sigma must be positive."
            )

        if self_loop_weight <= 0.0:
            raise ValueError(
                "self_loop_weight must be positive."
            )

        if eps <= 0.0:
            raise ValueError(
                "eps must be positive."
            )

        weights = {
            "spatial": spatial_weight,
            "visual": visual_weight,
            "attention": attention_weight,
            "overlap": overlap_weight,
            "size": size_weight,
            "shared_region": (
                shared_region_weight
            ),
            "relevance": relevance_weight,
        }

        if any(
            float(value) < 0.0
            for value in weights.values()
        ):
            raise ValueError(
                "Graph component weights must "
                "be non-negative."
            )

        weight_sum = sum(
            float(value)
            for value in weights.values()
        )

        if weight_sum <= 0.0:
            raise ValueError(
                "At least one graph weight "
                "must be positive."
            )

        self.top_k = int(top_k)
        self.minimum_neighbors = int(
            minimum_neighbors
        )
        self.minimum_edge_score = float(
            minimum_edge_score
        )
        self.spatial_sigma = float(
            spatial_sigma
        )
        self.self_loop_weight = float(
            self_loop_weight
        )
        self.eps = float(eps)

        self.component_weights = {
            name: float(value) / weight_sum
            for name, value
            in weights.items()
        }

        self.relation_dimension = 17

    @staticmethod
    def _pairwise_cosine(features):
        flattened = features.reshape(
            features.shape[0],
            -1,
        ).float()

        normalized = F.normalize(
            flattened,
            p=2,
            dim=-1,
            eps=1e-12,
        )

        cosine = torch.matmul(
            normalized,
            normalized.transpose(0, 1),
        ).clamp(
            min=-1.0,
            max=1.0,
        )

        return 0.5 * (cosine + 1.0)

    @staticmethod
    def _validate_nodes(nodes):
        if not nodes:
            raise ValueError(
                "At least one node is required."
            )

        raw_shapes = {
            tuple(
                node.raw_node_feature.shape
            )
            for node in nodes
        }
        profile_shapes = {
            tuple(
                node.attention_profile.shape
            )
            for node in nodes
        }

        if len(raw_shapes) != 1:
            raise ValueError(
                "All raw node features must "
                "have the same shape."
            )

        if len(profile_shapes) != 1:
            raise ValueError(
                "All Attention Profiles must "
                "have the same shape."
            )

        expected_ids = tuple(
            range(len(nodes))
        )
        actual_ids = tuple(
            int(node.node_id)
            for node in nodes
        )

        if actual_ids != expected_ids:
            raise ValueError(
                "Node IDs must be sequential."
            )

        for node in nodes:
            if not bool(
                torch.isfinite(
                    node.raw_node_feature
                ).all()
            ):
                raise ValueError(
                    "A raw node feature contains "
                    "NaN or Inf."
                )

            if not bool(
                torch.isfinite(
                    node.attention_profile
                ).all()
            ):
                raise ValueError(
                    "An Attention Profile contains "
                    "NaN or Inf."
                )

    def _build_constant_matrices(
        self,
        nodes,
        device,
    ):
        node_count = len(nodes)

        centers = torch.tensor(
            [
                node.normalized_center
                for node in nodes
            ],
            device=device,
            dtype=torch.float32,
        )

        widths = torch.tensor(
            [
                max(
                    node.normalized_xyxy[2]
                    - node.normalized_xyxy[0],
                    self.eps,
                )
                for node in nodes
            ],
            device=device,
            dtype=torch.float32,
        )

        heights = torch.tensor(
            [
                max(
                    node.normalized_xyxy[3]
                    - node.normalized_xyxy[1],
                    self.eps,
                )
                for node in nodes
            ],
            device=device,
            dtype=torch.float32,
        )

        areas = (widths * heights).clamp_min(
            self.eps
        )

        center_difference = (
            centers.unsqueeze(1)
            - centers.unsqueeze(0)
        )

        center_distance = torch.linalg.vector_norm(
            center_difference,
            dim=-1,
        )

        spatial_similarity = torch.exp(
            -center_distance.square()
            / (
                2.0
                * self.spatial_sigma ** 2
            )
        )

        log_areas = torch.log(areas)

        size_similarity = torch.exp(
            -torch.abs(
                log_areas.unsqueeze(1)
                - log_areas.unsqueeze(0)
            )
        )

        overlap_iou = torch.zeros(
            node_count,
            node_count,
            device=device,
            dtype=torch.float32,
        )

        same_class = torch.zeros_like(
            overlap_iou
        )
        shared_region = torch.zeros_like(
            overlap_iou
        )

        for first_index in range(node_count):
            first_regions = set(
                nodes[
                    first_index
                ].attention_region_ids
            )

            for second_index in range(
                first_index + 1,
                node_count,
            ):
                iou = _box_iou(
                    nodes[
                        first_index
                    ].normalized_xyxy,
                    nodes[
                        second_index
                    ].normalized_xyxy,
                )

                overlap_iou[
                    first_index,
                    second_index,
                ] = iou
                overlap_iou[
                    second_index,
                    first_index,
                ] = iou

                class_match = float(
                    nodes[first_index].class_id
                    >= 0
                    and (
                        nodes[first_index].class_id
                        == nodes[
                            second_index
                        ].class_id
                    )
                )

                same_class[
                    first_index,
                    second_index,
                ] = class_match
                same_class[
                    second_index,
                    first_index,
                ] = class_match

                second_regions = set(
                    nodes[
                        second_index
                    ].attention_region_ids
                )

                shared = float(
                    bool(
                        first_regions
                        .intersection(
                            second_regions
                        )
                    )
                )

                shared_region[
                    first_index,
                    second_index,
                ] = shared
                shared_region[
                    second_index,
                    first_index,
                ] = shared

        selection_scores = torch.tensor(
            [
                min(
                    max(
                        float(
                            node.selection_score
                        ),
                        0.0,
                    ),
                    1.0,
                )
                for node in nodes
            ],
            device=device,
            dtype=torch.float32,
        )

        relevance_similarity = torch.sqrt(
            (
                selection_scores.unsqueeze(1)
                * selection_scores.unsqueeze(0)
            ).clamp_min(0.0)
        )

        return {
            "centers": centers,
            "widths": widths,
            "heights": heights,
            "areas": areas,
            "center_distance": center_distance,
            "spatial_similarity": (
                spatial_similarity
            ),
            "size_similarity": size_similarity,
            "overlap_iou": overlap_iou,
            "same_class": same_class,
            "shared_region": shared_region,
            "selection_scores": (
                selection_scores
            ),
            "relevance_similarity": (
                relevance_similarity
            ),
        }

    def _select_edge_pairs(
        self,
        dense_scores,
    ):
        node_count = dense_scores.shape[0]

        if node_count == 1:
            return tuple()

        detached_scores = (
            dense_scores
            .detach()
            .to("cpu")
        )

        selected_pairs = set()

        for node_index in range(node_count):
            candidates = [
                (
                    float(
                        detached_scores[
                            node_index,
                            neighbor_index,
                        ].item()
                    ),
                    neighbor_index,
                )
                for neighbor_index
                in range(node_count)
                if neighbor_index != node_index
            ]

            candidates.sort(
                key=lambda item: (
                    -item[0],
                    item[1],
                )
            )

            selected_neighbors = [
                neighbor_index
                for score, neighbor_index
                in candidates
                if score
                >= self.minimum_edge_score
            ][:self.top_k]

            if len(selected_neighbors) < (
                self.minimum_neighbors
            ):
                already_selected = set(
                    selected_neighbors
                )

                for _, neighbor_index in (
                    candidates
                ):
                    if (
                        neighbor_index
                        in already_selected
                    ):
                        continue

                    selected_neighbors.append(
                        neighbor_index
                    )
                    already_selected.add(
                        neighbor_index
                    )

                    if len(
                        selected_neighbors
                    ) >= self.minimum_neighbors:
                        break

            for neighbor_index in (
                selected_neighbors
            ):
                selected_pairs.add(
                    tuple(
                        sorted(
                            (
                                node_index,
                                neighbor_index,
                            )
                        )
                    )
                )

        return tuple(
            sorted(selected_pairs)
        )

    def _directed_relation_feature(
        self,
        source_index,
        target_index,
        nodes,
        matrices,
        visual_similarity,
        attention_similarity,
        dense_scores,
    ):
        device = dense_scores.device

        source_center = matrices[
            "centers"
        ][source_index]
        target_center = matrices[
            "centers"
        ][target_index]

        delta = (
            target_center - source_center
        )

        width_ratio = torch.log(
            matrices["widths"][target_index]
            / matrices["widths"][source_index]
        ).clamp(
            min=-4.0,
            max=4.0,
        ) / 4.0

        height_ratio = torch.log(
            matrices["heights"][target_index]
            / matrices["heights"][source_index]
        ).clamp(
            min=-4.0,
            max=4.0,
        ) / 4.0

        same_class = float(
            nodes[source_index].class_id >= 0
            and (
                nodes[source_index].class_id
                == nodes[target_index].class_id
            )
        )

        source_attention_mass = min(
            max(
                float(
                    nodes[
                        source_index
                    ].attention_mass
                ),
                0.0,
            ),
            1.0,
        )
        target_attention_mass = min(
            max(
                float(
                    nodes[
                        target_index
                    ].attention_mass
                ),
                0.0,
            ),
            1.0,
        )

        values = [
            delta[0],
            delta[1],
            matrices[
                "center_distance"
            ][source_index, target_index],
            width_ratio,
            height_ratio,
            matrices[
                "overlap_iou"
            ][source_index, target_index],
            matrices[
                "spatial_similarity"
            ][source_index, target_index],
            visual_similarity[
                source_index,
                target_index,
            ],
            attention_similarity[
                source_index,
                target_index,
            ],
            matrices[
                "size_similarity"
            ][source_index, target_index],
            torch.tensor(
                same_class,
                device=device,
                dtype=torch.float32,
            ),
            matrices[
                "shared_region"
            ][source_index, target_index],
            matrices[
                "selection_scores"
            ][source_index],
            matrices[
                "selection_scores"
            ][target_index],
            torch.tensor(
                source_attention_mass,
                device=device,
                dtype=torch.float32,
            ),
            torch.tensor(
                target_attention_mass,
                device=device,
                dtype=torch.float32,
            ),
            dense_scores[
                source_index,
                target_index,
            ],
        ]

        return torch.stack(values)

    def forward(
        self,
        nodes: Sequence[HybridGraphNode],
    ) -> SparseObjectGraphOutput:
        nodes = tuple(nodes)

        self._validate_nodes(nodes)

        device = (
            nodes[0]
            .raw_node_feature
            .device
        )

        node_features = torch.stack(
            [
                node.raw_node_feature
                .to(device)
                .float()
                for node in nodes
            ],
            dim=0,
        )

        attention_profiles = torch.stack(
            [
                node.attention_profile
                .to(device)
                .float()
                for node in nodes
            ],
            dim=0,
        )

        visual_features = torch.stack(
            [
                node.visual_feature
                .to(device)
                .float()
                for node in nodes
            ],
            dim=0,
        )

        visual_similarity = (
            self._pairwise_cosine(
                visual_features
            )
        )
        attention_similarity = (
            self._pairwise_cosine(
                attention_profiles
            )
        )

        matrices = (
            self._build_constant_matrices(
                nodes=nodes,
                device=device,
            )
        )

        weights = self.component_weights

        dense_scores = (
            weights["spatial"]
            * matrices[
                "spatial_similarity"
            ]
            + weights["visual"]
            * visual_similarity
            + weights["attention"]
            * attention_similarity
            + weights["overlap"]
            * matrices["overlap_iou"]
            + weights["size"]
            * matrices["size_similarity"]
            + weights["shared_region"]
            * matrices["shared_region"]
            + weights["relevance"]
            * matrices[
                "relevance_similarity"
            ]
        ).clamp(
            min=0.0,
            max=1.0,
        )

        dense_scores = (
            dense_scores
            - torch.diag_embed(
                torch.diagonal(
                    dense_scores
                )
            )
        )

        edge_pairs = self._select_edge_pairs(
            dense_scores
        )

        node_count = len(nodes)

        sparse_adjacency = torch.zeros(
            node_count,
            node_count,
            device=device,
            dtype=torch.float32,
        )
        edge_mask = torch.zeros(
            node_count,
            node_count,
            device=device,
            dtype=torch.bool,
        )

        edges = []
        directed_sources = []
        directed_targets = []
        directed_weights = []
        directed_features = []

        for edge_id, (
            first_index,
            second_index,
        ) in enumerate(edge_pairs):
            weight = dense_scores[
                first_index,
                second_index,
            ]

            sparse_adjacency[
                first_index,
                second_index,
            ] = weight
            sparse_adjacency[
                second_index,
                first_index,
            ] = weight

            edge_mask[
                first_index,
                second_index,
            ] = True
            edge_mask[
                second_index,
                first_index,
            ] = True

            forced = bool(
                float(
                    weight
                    .detach()
                    .cpu()
                    .item()
                )
                < self.minimum_edge_score
            )

            edges.append(
                SparseGraphEdge(
                    edge_id=edge_id,
                    first_node=first_index,
                    second_node=second_index,
                    weight=float(
                        weight
                        .detach()
                        .cpu()
                        .item()
                    ),
                    forced=forced,
                    spatial_similarity=float(
                        matrices[
                            "spatial_similarity"
                        ][
                            first_index,
                            second_index,
                        ]
                        .detach()
                        .cpu()
                        .item()
                    ),
                    visual_similarity=float(
                        visual_similarity[
                            first_index,
                            second_index,
                        ]
                        .detach()
                        .cpu()
                        .item()
                    ),
                    attention_similarity=float(
                        attention_similarity[
                            first_index,
                            second_index,
                        ]
                        .detach()
                        .cpu()
                        .item()
                    ),
                    overlap_iou=float(
                        matrices[
                            "overlap_iou"
                        ][
                            first_index,
                            second_index,
                        ]
                        .detach()
                        .cpu()
                        .item()
                    ),
                    size_similarity=float(
                        matrices[
                            "size_similarity"
                        ][
                            first_index,
                            second_index,
                        ]
                        .detach()
                        .cpu()
                        .item()
                    ),
                    shared_attention_region=(
                        float(
                            matrices[
                                "shared_region"
                            ][
                                first_index,
                                second_index,
                            ]
                            .detach()
                            .cpu()
                            .item()
                        )
                    ),
                    relevance_similarity=float(
                        matrices[
                            "relevance_similarity"
                        ][
                            first_index,
                            second_index,
                        ]
                        .detach()
                        .cpu()
                        .item()
                    ),
                )
            )

            for source, target in (
                (
                    first_index,
                    second_index,
                ),
                (
                    second_index,
                    first_index,
                ),
            ):
                directed_sources.append(source)
                directed_targets.append(target)
                directed_weights.append(weight)
                directed_features.append(
                    self._directed_relation_feature(
                        source_index=source,
                        target_index=target,
                        nodes=nodes,
                        matrices=matrices,
                        visual_similarity=(
                            visual_similarity
                        ),
                        attention_similarity=(
                            attention_similarity
                        ),
                        dense_scores=dense_scores,
                    )
                )

        if directed_sources:
            edge_index = torch.tensor(
                [
                    directed_sources,
                    directed_targets,
                ],
                device=device,
                dtype=torch.long,
            )
            edge_weight = torch.stack(
                directed_weights
            )
            edge_features = torch.stack(
                directed_features,
                dim=0,
            )
        else:
            edge_index = torch.empty(
                2,
                0,
                device=device,
                dtype=torch.long,
            )
            edge_weight = torch.empty(
                0,
                device=device,
                dtype=torch.float32,
            )
            edge_features = torch.empty(
                0,
                self.relation_dimension,
                device=device,
                dtype=torch.float32,
            )

        degree = edge_mask.sum(
            dim=-1
        ).to(torch.long)

        adjacency_with_self = (
            sparse_adjacency
            + torch.eye(
                node_count,
                device=device,
                dtype=torch.float32,
            )
            * self.self_loop_weight
        )

        normalized_adjacency = (
            adjacency_with_self
            / adjacency_with_self.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(self.eps)
        )

        output_tensors = (
            node_features,
            edge_weight,
            edge_features,
            dense_scores,
            sparse_adjacency,
            normalized_adjacency,
            matrices["spatial_similarity"],
            visual_similarity,
            attention_similarity,
            matrices["overlap_iou"],
            matrices["size_similarity"],
            matrices["shared_region"],
            matrices["relevance_similarity"],
        )

        if not all(
            bool(torch.isfinite(tensor).all())
            for tensor in output_tensors
        ):
            raise RuntimeError(
                "Sparse Graph contains NaN or Inf."
            )

        return SparseObjectGraphOutput(
            node_features=node_features,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_features=edge_features,
            dense_edge_scores=dense_scores,
            sparse_adjacency=(
                sparse_adjacency
            ),
            normalized_adjacency=(
                normalized_adjacency
            ),
            edge_mask=edge_mask,
            degree=degree,
            spatial_similarity=matrices[
                "spatial_similarity"
            ],
            visual_similarity=(
                visual_similarity
            ),
            attention_similarity=(
                attention_similarity
            ),
            overlap_iou=matrices[
                "overlap_iou"
            ],
            size_similarity=matrices[
                "size_similarity"
            ],
            shared_attention_region=matrices[
                "shared_region"
            ],
            relevance_similarity=matrices[
                "relevance_similarity"
            ],
            edges=tuple(edges),
            relation_dimension=(
                self.relation_dimension
            ),
            undirected_edge_count=len(
                edge_pairs
            ),
            directed_edge_count=len(
                directed_sources
            ),
        )