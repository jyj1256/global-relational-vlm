import argparse
import math
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from vlm.models.graph_readout import GraphTokenHead
from vlm.models.weighted_gnn import WeightedGNN


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test Graph Readout and projection."
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
        "--output",
        type=Path,
        default=Path(
            "/home/dm-magneto/dm_yunjin/"
            "vlm/outputs/graph_visualizations/candy/"
            "untrained_graph_token_test.pt"
        ),
    )
    parser.add_argument(
        "--graph-dim",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--llm-hidden-dim",
        type=int,
        default=3584,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


def gradient_norm(modules):
    squared_sum = 0.0

    for module in modules:
        for parameter in module.parameters():
            if parameter.grad is not None:
                squared_sum += float(
                    parameter.grad
                    .float()
                    .pow(2)
                    .sum()
                )

    return math.sqrt(squared_sum)


def count_parameters(module):
    return sum(
        parameter.numel()
        for parameter in module.parameters()
    )


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

    gnn = WeightedGNN(
        input_dim=node_features.shape[-1],
        hidden_dim=args.graph_dim,
        num_layers=2,
        use_layer_norm=True,
    )
    graph_head = GraphTokenHead(
        graph_dim=args.graph_dim,
        llm_hidden_dim=args.llm_hidden_dim,
        use_layer_norm=True,
    )

    gnn.train()
    graph_head.train()

    gnn_output = gnn(
        node_features=node_features,
        normalized_edge_weights=edge_weights,
    )

    token_output = graph_head(
        gnn_output.final_node_features
    )

    expected_mean = (
        gnn_output.final_node_features
        .mean(dim=0, keepdim=True)
    )

    if not torch.allclose(
        token_output.graph_embedding,
        expected_mean,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Mean Readout result is incorrect."
        )

    test_loss = (
        token_output.graph_token
        .pow(2)
        .mean()
    )
    test_loss.backward()

    gnn_parameters = count_parameters(gnn)
    head_parameters = count_parameters(
        graph_head
    )
    total_parameters = (
        gnn_parameters + head_parameters
    )
    total_grad_norm = gradient_norm(
        (gnn, graph_head)
    )

    print("Graph Readout summary:")
    print(
        f"  GNN node features: "
        f"{tuple(gnn_output.final_node_features.shape)}"
    )
    print(
        f"  Graph embedding: "
        f"{tuple(token_output.graph_embedding.shape)}"
    )
    print(
        f"  LLM Graph token: "
        f"{tuple(token_output.graph_token.shape)}"
    )
    print(
        f"  Readout: Mean"
    )

    print("\nParameter counts:")
    print(
        f"  Weighted GNN: "
        f"{gnn_parameters:,}"
    )
    print(
        f"  Graph head: "
        f"{head_parameters:,}"
    )
    print(
        f"  Total: "
        f"{total_parameters:,}"
    )

    print("\nNumerical checks:")
    print(
        f"  Graph embedding norm: "
        f"{token_output.graph_embedding.norm().item():.6f}"
    )
    print(
        f"  Graph token norm: "
        f"{token_output.graph_token.norm().item():.6f}"
    )
    print(
        f"  Graph token mean/std: "
        f"{token_output.graph_token.mean().item():.6f} / "
        f"{token_output.graph_token.std().item():.6f}"
    )
    print(
        f"  NaN: "
        f"{torch.isnan(token_output.graph_token).any().item()}"
    )
    print(
        f"  Inf: "
        f"{torch.isinf(token_output.graph_token).any().item()}"
    )
    print(
        f"  Test loss: "
        f"{test_loss.item():.8f}"
    )
    print(
        f"  Gradient norm: "
        f"{total_grad_norm:.8f}"
    )

    if token_output.graph_embedding.shape != (
        1,
        args.graph_dim,
    ):
        raise RuntimeError(
            "Unexpected Graph embedding shape."
        )

    if token_output.graph_token.shape != (
        1,
        args.llm_hidden_dim,
    ):
        raise RuntimeError(
            "Unexpected LLM Graph token shape."
        )

    if total_grad_norm <= 0.0:
        raise RuntimeError(
            "Graph modules received no gradients."
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "graph_embedding": (
                token_output.graph_embedding
                .detach()
                .cpu()
            ),
            "graph_token": (
                token_output.graph_token
                .detach()
                .cpu()
            ),
            "gnn_state_dict": (
                gnn.state_dict()
            ),
            "graph_head_state_dict": (
                graph_head.state_dict()
            ),
            "metadata": {
                "trained": False,
                "readout": "mean",
                "graph_dim": args.graph_dim,
                "llm_hidden_dim": (
                    args.llm_hidden_dim
                ),
                "seed": args.seed,
            },
        },
        args.output,
    )

    print(
        f"\nSaved untrained test artifact: "
        f"{args.output}"
    )
    print("\nGraph Readout test: PASS")


if __name__ == "__main__":
    main()