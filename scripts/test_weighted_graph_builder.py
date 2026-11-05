import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from vlm.models.weighted_graph_builder import (
    build_weighted_graph,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test weighted salient-region graph."
    )
    parser.add_argument(
        "--nodes",
        type=Path,
        default=Path(
            "/home/dm-magneto/dm_yunjin/"
            "vlm/outputs/attention_maps/candy/"
            "graph_nodes_top16.pt"
        ),
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path(
            "/home/dm-magneto/dm_yunjin/"
            "vlm/outputs/attention_maps/candy/"
            "local_attention_profiles.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/dm-magneto/dm_yunjin/"
            "vlm/outputs/graph_visualizations/candy/"
            "weighted_graph_top16.pt"
        ),
    )
    parser.add_argument(
        "--sigma-p",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
    )
    return parser.parse_args()


def off_diagonal_values(matrix):
    mask = ~torch.eye(
        matrix.shape[0],
        dtype=torch.bool,
    )
    return matrix[mask]


def print_range(name, matrix):
    values = off_diagonal_values(matrix)

    print(
        f"  {name}: "
        f"min={values.min().item():.6f}, "
        f"mean={values.mean().item():.6f}, "
        f"max={values.max().item():.6f}"
    )


def main():
    args = parse_args()

    node_data = torch.load(
        args.nodes,
        map_location="cpu",
        weights_only=True,
    )
    profile_data = torch.load(
        args.profiles,
        map_location="cpu",
        weights_only=True,
    )

    local_token_indices = (
        node_data["local_token_indices"].long()
    )
    all_profiles = profile_data["profiles"]

    selected_profiles = (
        all_profiles
        .index_select(
            dim=0,
            index=local_token_indices,
        )
        .flatten(start_dim=1)
    )

    graph = build_weighted_graph(
        visual_features=(
            node_data["visual_features"]
        ),
        normalized_positions=(
            node_data["normalized_positions"]
        ),
        attention_scores=(
            node_data["attention_scores"]
        ),
        attention_profiles=selected_profiles,
        sigma_p=args.sigma_p,
        threshold=args.threshold,
    )

    threshold_mask = (
        graph.dense_edge_weights
        >= args.threshold
    )
    threshold_mask.fill_diagonal_(False)

    retained_upper = torch.triu(
        graph.retained_edge_mask,
        diagonal=1,
    )
    threshold_upper = torch.triu(
        threshold_mask,
        diagonal=1,
    )

    retained_edges = int(
        retained_upper.sum().item()
    )
    threshold_edges = int(
        threshold_upper.sum().item()
    )
    forced_edges = int(
        (
            retained_upper
            & ~threshold_upper
        ).sum().item()
    )

    degrees = (
        graph.retained_edge_mask
        .sum(dim=1)
        .float()
    )
    row_sums = (
        graph.normalized_edge_weights
        .sum(dim=1)
    )

    print("Weighted graph summary:")
    print(
        f"  Node features: "
        f"{tuple(graph.node_features.shape)}"
    )
    print(
        f"  Attention profiles: "
        f"{tuple(selected_profiles.shape)}"
    )
    print(f"  Sigma-p: {args.sigma_p}")
    print(f"  Threshold: {args.threshold}")

    print("\nPairwise score ranges:")
    print_range(
        "Spatial similarity",
        graph.spatial_similarity,
    )
    print_range(
        "Visual similarity",
        graph.visual_similarity,
    )
    print_range(
        "Attention similarity",
        graph.attention_similarity,
    )
    print_range(
        "Dense edge weight",
        graph.dense_edge_weights,
    )

    print("\nPruning summary:")
    print(
        f"  Edges above threshold: "
        f"{threshold_edges}"
    )
    print(
        f"  Retained undirected edges: "
        f"{retained_edges}"
    )
    print(
        f"  Forced fallback edges: "
        f"{forced_edges}"
    )
    print(
        f"  Degree min/mean/max: "
        f"{degrees.min().item():.0f}/"
        f"{degrees.mean().item():.2f}/"
        f"{degrees.max().item():.0f}"
    )
    print(
        f"  Normalized row-sum range: "
        f"{row_sums.min().item():.8f} - "
        f"{row_sums.max().item():.8f}"
    )

    if not torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-6,
    ):
        raise RuntimeError(
            "Normalized edge rows do not sum to 1."
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "node_features": graph.node_features,
            "attention_profiles": (
                graph.attention_profiles
            ),
            "spatial_similarity": (
                graph.spatial_similarity
            ),
            "visual_similarity": (
                graph.visual_similarity
            ),
            "attention_similarity": (
                graph.attention_similarity
            ),
            "dense_edge_weights": (
                graph.dense_edge_weights
            ),
            "retained_edge_mask": (
                graph.retained_edge_mask
            ),
            "pruned_edge_weights": (
                graph.pruned_edge_weights
            ),
            "normalized_edge_weights": (
                graph.normalized_edge_weights
            ),
            "local_token_indices": (
                local_token_indices
            ),
            "sigma_p": args.sigma_p,
            "threshold": args.threshold,
            "coefficients": {
                "spatial": 0.35,
                "visual": 0.20,
                "attention": 0.45,
            },
        },
        args.output,
    )

    print(f"\nSaved weighted graph: {args.output}")
    print("\nWeighted graph builder test: PASS")


if __name__ == "__main__":
    main()