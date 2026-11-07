import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from vlm.models.object_graph_v1.sparse_graph_builder import (
    SparseObjectGraphBuilder,
)


ARTIFACT_PATH = (
    PROJECT_ROOT
    / "vlm/outputs/"
    "object_graph_visualizations/"
    "baseball_2337002/"
    "hybrid_nodes_test.pt"
)


def main():
    if not ARTIFACT_PATH.is_file():
        raise FileNotFoundError(
            ARTIFACT_PATH
        )

    artifact = torch.load(
        ARTIFACT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    nodes = list(
        artifact["nodes"]
    )

    if not nodes:
        raise RuntimeError(
            "Hybrid Node artifact is empty."
        )

    # Loaded tensors are converted into fresh leaves
    # so gradient propagation can be tested.
    for node in nodes:
        node.visual_feature = (
            node.visual_feature
            .detach()
            .clone()
            .requires_grad_(True)
        )
        node.attention_profile = (
            node.attention_profile
            .detach()
            .clone()
            .requires_grad_(True)
        )
        node.geometry_feature = (
            node.geometry_feature
            .detach()
            .clone()
        )
        node.scalar_feature = (
            node.scalar_feature
            .detach()
            .clone()
        )

    builder = SparseObjectGraphBuilder(
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
    )

    output = builder(nodes)

    node_count = len(nodes)
    complete_edge_count = (
        node_count * (node_count - 1) // 2
    )

    row_sums = (
        output.normalized_adjacency.sum(
            dim=-1
        )
    )

    symmetric_matrices = (
        output.dense_edge_scores,
        output.sparse_adjacency,
        output.spatial_similarity,
        output.visual_similarity,
        output.attention_similarity,
        output.overlap_iou,
        output.size_similarity,
        output.shared_attention_region,
        output.relevance_similarity,
    )

    matrices_symmetric = all(
        bool(
            torch.allclose(
                matrix,
                matrix.transpose(0, 1),
                atol=1e-6,
                rtol=0.0,
            )
        )
        for matrix in symmetric_matrices
    )

    adjacency_matches_edges = True

    for edge_position in range(
        output.directed_edge_count
    ):
        source = int(
            output.edge_index[
                0,
                edge_position,
            ].item()
        )
        target = int(
            output.edge_index[
                1,
                edge_position,
            ].item()
        )

        expected_weight = (
            output.sparse_adjacency[
                source,
                target,
            ]
        )

        if not bool(
            torch.allclose(
                output.edge_weight[
                    edge_position
                ],
                expected_weight,
                atol=1e-6,
                rtol=0.0,
            )
        ):
            adjacency_matches_edges = False
            break

    reverse_edges_valid = True
    directional_relations_valid = True

    symmetric_relation_indices = (
        2,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        16,
    )
    signed_relation_indices = (
        0,
        1,
        3,
        4,
    )

    for edge_position in range(
        0,
        output.directed_edge_count,
        2,
    ):
        forward_index = (
            output.edge_index[
                :,
                edge_position,
            ]
        )
        reverse_index = (
            output.edge_index[
                :,
                edge_position + 1,
            ]
        )

        if not bool(
            torch.equal(
                forward_index,
                reverse_index.flip(0),
            )
        ):
            reverse_edges_valid = False

        forward_feature = (
            output.edge_features[
                edge_position
            ]
        )
        reverse_feature = (
            output.edge_features[
                edge_position + 1
            ]
        )

        for relation_index in (
            signed_relation_indices
        ):
            if not bool(
                torch.allclose(
                    forward_feature[
                        relation_index
                    ],
                    -reverse_feature[
                        relation_index
                    ],
                    atol=1e-6,
                    rtol=0.0,
                )
            ):
                directional_relations_valid = (
                    False
                )

        for relation_index in (
            symmetric_relation_indices
        ):
            if not bool(
                torch.allclose(
                    forward_feature[
                        relation_index
                    ],
                    reverse_feature[
                        relation_index
                    ],
                    atol=1e-6,
                    rtol=0.0,
                )
            ):
                directional_relations_valid = (
                    False
                )

        if not bool(
            torch.allclose(
                forward_feature[12],
                reverse_feature[13],
                atol=1e-6,
                rtol=0.0,
            )
            and torch.allclose(
                forward_feature[13],
                reverse_feature[12],
                atol=1e-6,
                rtol=0.0,
            )
            and torch.allclose(
                forward_feature[14],
                reverse_feature[15],
                atol=1e-6,
                rtol=0.0,
            )
            and torch.allclose(
                forward_feature[15],
                reverse_feature[14],
                atol=1e-6,
                rtol=0.0,
            )
        ):
            directional_relations_valid = (
                False
            )

    correct_bat_nodes = [
        node
        for node in nodes
        if (
            node.class_name
            == "baseball bat"
            and node.detection_id == 4
        )
    ]

    correct_bat_connected = False

    if len(correct_bat_nodes) == 1:
        bat_node_id = (
            correct_bat_nodes[0].node_id
        )
        correct_bat_connected = (
            int(
                output.degree[
                    bat_node_id
                ].item()
            )
            >= 1
        )

    finite_tensors = (
        output.node_features,
        output.edge_weight,
        output.edge_features,
        output.dense_edge_scores,
        output.sparse_adjacency,
        output.normalized_adjacency,
    )

    finite_outputs = all(
        bool(torch.isfinite(tensor).all())
        for tensor in finite_tensors
    )

    loss = (
        output.node_features
        .square()
        .mean()
        + output.edge_weight
        .square()
        .mean()
        + output.edge_features
        .square()
        .mean()
        + output.normalized_adjacency
        .square()
        .mean()
    )

    loss.backward()

    visual_gradient_norm = sum(
        float(
            node.visual_feature.grad
            .norm()
            .item()
        )
        for node in nodes
        if node.visual_feature.grad
        is not None
    )

    profile_gradient_norm = sum(
        float(
            node.attention_profile.grad
            .norm()
            .item()
        )
        for node in nodes
        if node.attention_profile.grad
        is not None
    )

    single_node_output = builder(
        nodes[:1]
    )

    checks = {
        "node_count": (
            output.node_features.shape[0]
            == node_count
        ),
        "node_feature_shape": (
            tuple(
                output.node_features.shape
            )
            == (
                node_count,
                nodes[0]
                .raw_node_feature
                .numel(),
            )
        ),
        "relation_dimension": (
            output.relation_dimension == 17
        ),
        "edge_feature_shape": (
            tuple(
                output.edge_features.shape
            )
            == (
                output.directed_edge_count,
                17,
            )
        ),
        "directed_edge_count": (
            output.directed_edge_count
            == 2
            * output.undirected_edge_count
        ),
        "sparse_not_complete": (
            output.undirected_edge_count
            < complete_edge_count
        ),
        "nonempty_edges": (
            output.undirected_edge_count > 0
        ),
        "minimum_degree": (
            int(output.degree.min().item())
            >= 1
        ),
        "adjacency_symmetric": (
            matrices_symmetric
        ),
        "edge_mask_symmetric": bool(
            torch.equal(
                output.edge_mask,
                output.edge_mask.transpose(
                    0,
                    1,
                ),
            )
        ),
        "zero_sparse_diagonal": bool(
            torch.allclose(
                torch.diagonal(
                    output.sparse_adjacency
                ),
                torch.zeros(
                    node_count,
                    dtype=torch.float32,
                ),
                atol=1e-7,
                rtol=0.0,
            )
        ),
        "normalized_rows": bool(
            torch.allclose(
                row_sums,
                torch.ones_like(row_sums),
                atol=1e-6,
                rtol=0.0,
            )
        ),
        "edge_weight_range": bool(
            (output.edge_weight > 0.0).all()
            and (
                output.edge_weight <= 1.0
            ).all()
        ),
        "adjacency_matches_edges": (
            adjacency_matches_edges
        ),
        "reverse_edges": (
            reverse_edges_valid
        ),
        "directional_relations": (
            directional_relations_valid
        ),
        "correct_bat_connected": (
            correct_bat_connected
        ),
        "finite_outputs": finite_outputs,
        "visual_gradient": (
            visual_gradient_norm > 0.0
        ),
        "profile_gradient": (
            profile_gradient_norm > 0.0
        ),
        "single_node_edge_count": (
            single_node_output
            .undirected_edge_count
            == 0
            and single_node_output
            .directed_edge_count
            == 0
        ),
        "single_node_identity": bool(
            torch.allclose(
                single_node_output
                .normalized_adjacency,
                torch.ones(
                    1,
                    1,
                    dtype=torch.float32,
                ),
                atol=1e-6,
                rtol=0.0,
            )
        ),
        "no_trainable_parameters": (
            sum(
                parameter.numel()
                for parameter
                in builder.parameters()
            )
            == 0
        ),
    }

    print(
        "Sparse Object Graph summary:"
    )
    print(
        "  Nodes:",
        node_count,
    )
    print(
        "  Complete graph edges:",
        complete_edge_count,
    )
    print(
        "  Retained undirected edges:",
        output.undirected_edge_count,
    )
    print(
        "  Directed edges:",
        output.directed_edge_count,
    )
    print(
        "  Relation dimension:",
        output.relation_dimension,
    )
    print(
        "  Degree min/mean/max:",
        (
            int(output.degree.min().item()),
            float(
                output.degree
                .float()
                .mean()
                .item()
            ),
            int(output.degree.max().item()),
        ),
    )
    print(
        "  Forced edges:",
        sum(
            int(edge.forced)
            for edge in output.edges
        ),
    )
    print(
        "  Normalized row sums:",
        (
            float(row_sums.min().item()),
            float(row_sums.max().item()),
        ),
    )

    print("\nRetained edges:")

    ranked_edges = sorted(
        output.edges,
        key=lambda edge: (
            -edge.weight,
            edge.edge_id,
        ),
    )

    for edge in ranked_edges:
        first_node = nodes[
            edge.first_node
        ]
        second_node = nodes[
            edge.second_node
        ]

        print(
            f"  E{edge.edge_id:02d} "
            f"N{edge.first_node:02d}"
            f"({first_node.class_name}) "
            f"<-> "
            f"N{edge.second_node:02d}"
            f"({second_node.class_name}) "
            f"w={edge.weight:.3f} "
            f"spatial="
            f"{edge.spatial_similarity:.3f} "
            f"visual="
            f"{edge.visual_similarity:.3f} "
            f"attention="
            f"{edge.attention_similarity:.3f} "
            f"IoU={edge.overlap_iou:.3f} "
            f"forced={edge.forced}"
        )

    print("\nGradient diagnostics:")
    print(
        "  Test loss:",
        f"{loss.item():.8f}",
    )
    print(
        "  Visual gradient norm:",
        f"{visual_gradient_norm:.8f}",
    )
    print(
        "  Profile gradient norm:",
        f"{profile_gradient_norm:.8f}",
    )

    print("\nChecks:")

    for name, passed in checks.items():
        print(
            f"  {name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    if not all(checks.values()):
        failed = [
            name
            for name, passed
            in checks.items()
            if not passed
        ]

        raise RuntimeError(
            "Sparse Object Graph test "
            f"failed: {failed}"
        )

    print(
        "\nSparse Object Graph test: PASS"
    )
    print(
        "Note: graph topology uses the real "
        "baseball node geometry and Attention "
        "statistics, but synthetic visual features."
    )


if __name__ == "__main__":
    main()