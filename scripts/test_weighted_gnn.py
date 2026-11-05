import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from vlm.models.weighted_gnn import WeightedGNN


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test the two-layer weighted GNN."
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path(
            "/home/dm-magneto/dm_yunjin/"
            "vlm/outputs/graph_visualizations/candy/"
            "weighted_graph_top16.pt"
        ),
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def off_diagonal_cosine_mean(features):
    normalized = F.normalize(
        features.float(),
        p=2,
        dim=-1,
        eps=1e-12,
    )
    similarity = normalized @ normalized.T

    mask = ~torch.eye(
        similarity.shape[0],
        dtype=torch.bool,
        device=similarity.device,
    )

    return float(similarity[mask].mean())


def total_gradient_norm(model):
    squared_sum = 0.0

    for parameter in model.parameters():
        if parameter.grad is not None:
            squared_sum += float(
                parameter.grad
                .float()
                .pow(2)
                .sum()
            )

    return math.sqrt(squared_sum)


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    graph_data = torch.load(
        args.graph,
        map_location="cpu",
        weights_only=True,
    )

    node_features = (
        graph_data["node_features"].float()
    )
    edge_weights = (
        graph_data[
            "normalized_edge_weights"
        ].float()
    )

    row_sums = edge_weights.sum(dim=1)

    if not torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-6,
    ):
        raise RuntimeError(
            "Input edge rows do not sum to 1."
        )

    model = WeightedGNN(
        input_dim=node_features.shape[-1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_layer_norm=True,
    )
    model.train()

    output = model(
        node_features=node_features,
        normalized_edge_weights=edge_weights,
    )

    final_features = (
        output.final_node_features
    )

    test_loss = final_features.pow(2).mean()
    test_loss.backward()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    gradient_norm = total_gradient_norm(model)

    print("Weighted GNN summary:")
    print(
        f"  Input nodes: "
        f"{tuple(node_features.shape)}"
    )
    print(
        f"  Edge matrix: "
        f"{tuple(edge_weights.shape)}"
    )
    print(
        f"  GNN layers: "
        f"{len(output.layer_states)}"
    )
    print(
        f"  Hidden dimension: "
        f"{args.hidden_dim}"
    )
    print(
        f"  Trainable parameters: "
        f"{parameter_count:,}"
    )

    print("\nLayer outputs:")

    for layer_index, layer_state in enumerate(
        output.layer_states,
        start=1,
    ):
        print(
            f"  Layer {layer_index}: "
            f"shape={tuple(layer_state.shape)}, "
            f"mean={layer_state.mean().item():.6f}, "
            f"std={layer_state.std().item():.6f}, "
            f"norm_mean="
            f"{layer_state.norm(dim=-1).mean().item():.6f}, "
            f"cosine_mean="
            f"{off_diagonal_cosine_mean(layer_state):.6f}"
        )

    print("\nOutput checks:")
    print(
        f"  Final shape: "
        f"{tuple(final_features.shape)}"
    )
    print(
        f"  NaN: "
        f"{torch.isnan(final_features).any().item()}"
    )
    print(
        f"  Inf: "
        f"{torch.isinf(final_features).any().item()}"
    )
    print(
        f"  Test loss: "
        f"{test_loss.item():.8f}"
    )
    print(
        f"  Gradient norm: "
        f"{gradient_norm:.8f}"
    )
    print(
        f"  Input cosine mean: "
        f"{off_diagonal_cosine_mean(node_features):.6f}"
    )
    print(
        f"  Final cosine mean: "
        f"{off_diagonal_cosine_mean(final_features):.6f}"
    )

    if final_features.shape != (
        node_features.shape[0],
        args.hidden_dim,
    ):
        raise RuntimeError(
            "Unexpected final GNN output shape."
        )

    if gradient_norm <= 0.0:
        raise RuntimeError(
            "GNN parameters received no gradients."
        )

    if not torch.isfinite(
        final_features
    ).all():
        raise RuntimeError(
            "Final features are not finite."
        )

    print("\nWeighted GNN test: PASS")


if __name__ == "__main__":
    main()