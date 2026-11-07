import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from vlm.models.object_graph_v1.gated_readout import (
    GatedGraphReadout,
)
from vlm.models.object_graph_v1.residual_gated_gnn import (
    ResidualGatedGNN,
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


def parameter_count(module):
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def gradient_norm(module):
    squared_norm = 0.0

    for parameter in module.parameters():
        if parameter.grad is None:
            continue

        squared_norm += float(
            parameter.grad
            .detach()
            .float()
            .square()
            .sum()
            .item()
        )

    return squared_norm ** 0.5


def main():
    if not ARTIFACT_PATH.is_file():
        raise FileNotFoundError(
            ARTIFACT_PATH
        )

    torch.manual_seed(2026)

    artifact = torch.load(
        ARTIFACT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    nodes = list(artifact["nodes"])

    visual_leaves = []
    profile_leaves = []

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

        visual_leaves.append(
            node.visual_feature
        )
        profile_leaves.append(
            node.attention_profile
        )

    graph_builder = (
        SparseObjectGraphBuilder(
            top_k=3,
            minimum_neighbors=1,
            minimum_edge_score=0.15,
            spatial_sigma=0.35,
        )
    )

    graph = graph_builder(nodes)

    gnn = ResidualGatedGNN(
        input_dimension=int(
            graph.node_features.shape[-1]
        ),
        relation_dimension=(
            graph.relation_dimension
        ),
        hidden_dimension=256,
        layer_count=2,
        relation_hidden_dimension=64,
        gate_hidden_dimension=64,
        dropout=0.0,
        initial_residual_gate=-2.0,
    )

    gnn.eval()

    gnn_output = gnn(
        node_features=graph.node_features,
        edge_index=graph.edge_index,
        edge_weight=graph.edge_weight,
        edge_features=graph.edge_features,
    )

    gnn_output.final_node_features.retain_grad()

    node_context = (
        torch.stack(
            [
                node.scalar_feature
                for node in nodes
            ],
            dim=0,
        )
        .detach()
        .clone()
        .requires_grad_(True)
    )

    node_prior = torch.tensor(
        [
            node.selection_score
            for node in nodes
        ],
        dtype=torch.float32,
    )

    readout = GatedGraphReadout(
        hidden_dimension=256,
        node_context_dimension=9,
        graph_context_dimension=128,
        context_hidden_dimension=64,
        score_hidden_dimension=128,
        gate_hidden_dimension=64,
        llm_hidden_dimension=3584,
        dropout=0.0,
        prior_strength=0.5,
        initial_readout_gate=-1.0,
        initial_token_scale=0.35,
        eps=1e-6,
    )

    readout.eval()

    output = readout(
        node_features=(
            gnn_output.final_node_features
        ),
        node_context=node_context,
        node_prior=node_prior,
    )

    repeated_output = readout(
        node_features=(
            gnn_output.final_node_features
        ),
        node_context=node_context,
        node_prior=node_prior,
    )

    repeat_error = float(
        (
            repeated_output.graph_token
            - output.graph_token
        )
        .abs()
        .max()
        .item()
    )

    node_count = len(nodes)

    permutation = torch.randperm(
        node_count
    )
    inverse_permutation = torch.empty_like(
        permutation
    )
    inverse_permutation[permutation] = (
        torch.arange(node_count)
    )

    permuted_output = readout(
        node_features=(
            gnn_output
            .final_node_features[
                permutation
            ]
        ),
        node_context=(
            node_context[permutation]
        ),
        node_prior=(
            node_prior[permutation]
        ),
    )

    permutation_token_error = float(
        (
            permuted_output.graph_token
            - output.graph_token
        )
        .abs()
        .max()
        .item()
    )

    restored_weights = (
        permuted_output.node_weights[
            inverse_permutation
        ]
    )

    permutation_weight_error = float(
        (
            restored_weights
            - output.node_weights
        )
        .abs()
        .max()
        .item()
    )

    correct_bat_indices = [
        index
        for index, node in enumerate(nodes)
        if (
            node.class_name
            == "baseball bat"
            and node.detection_id == 4
        )
    ]

    if len(correct_bat_indices) != 1:
        raise RuntimeError(
            "The correct baseball bat node "
            "was not found exactly once."
        )

    correct_bat_index = (
        correct_bat_indices[0]
    )

    focused_prior = torch.full(
        (node_count,),
        0.01,
        dtype=torch.float32,
    )
    focused_prior[
        correct_bat_index
    ] = 1.0

    focused_output = readout(
        node_features=(
            gnn_output.final_node_features
        ),
        node_context=node_context,
        node_prior=focused_prior,
    )

    focused_bat_weight = float(
        focused_output.node_weights[
            correct_bat_index
        ].item()
    )
    baseline_bat_weight = float(
        output.node_weights[
            correct_bat_index
        ].item()
    )

    single_output = readout(
        node_features=(
            gnn_output
            .final_node_features[:1]
        ),
        node_context=node_context[:1],
        node_prior=node_prior[:1],
    )

    normalized_rms = float(
        torch.sqrt(
            output
            .normalized_projected_feature
            .square()
            .mean()
        ).item()
    )

    token_scale = float(
        output.token_scale.item()
    )
    token_norm = float(
        output.graph_token
        .float()
        .norm()
        .item()
    )
    expected_token_norm = (
        token_scale
        * math.sqrt(3584)
    )
    token_norm_error = abs(
        token_norm
        - expected_token_norm
    )

    normalized_entropy = float(
        (
            -output.node_weights
            * torch.log(
                output.node_weights
                .clamp_min(1e-12)
            )
        )
        .sum()
        .div(math.log(node_count))
        .item()
    )

    prior_centered = (
        node_prior - node_prior.mean()
    )
    weight_centered = (
        output.node_weights
        - output.node_weights.mean()
    )

    prior_weight_correlation = float(
        (
            prior_centered
            * weight_centered
        )
        .sum()
        .div(
            prior_centered.norm()
            * weight_centered.norm()
            + 1e-12
        )
        .item()
    )

    target = torch.linspace(
        -0.1,
        0.1,
        3584,
        dtype=torch.float32,
    ).reshape(
        1,
        1,
        3584,
    )

    loss = F.mse_loss(
        output.graph_token,
        target,
    )

    loss.backward()

    visual_gradient_norm = sum(
        float(
            tensor.grad
            .detach()
            .float()
            .norm()
            .item()
        )
        for tensor in visual_leaves
        if tensor.grad is not None
    )

    profile_gradient_norm = sum(
        float(
            tensor.grad
            .detach()
            .float()
            .norm()
            .item()
        )
        for tensor in profile_leaves
        if tensor.grad is not None
    )

    module_gradients = {
        "gnn": gradient_norm(gnn),
        "node_norm": gradient_norm(
            readout.node_norm
        ),
        "context_encoder": gradient_norm(
            readout.context_encoder
        ),
        "graph_context": gradient_norm(
            readout.graph_context_encoder
        ),
        "attention_score": gradient_norm(
            readout.attention_score
        ),
        "node_gate": gradient_norm(
            readout.node_gate
        ),
        "readout_candidate": gradient_norm(
            readout.readout_candidate
        ),
        "readout_gate": gradient_norm(
            readout.readout_gate
        ),
        "graph_embedding_norm": gradient_norm(
            readout.graph_embedding_norm
        ),
        "token_projection": gradient_norm(
            readout.token_projection
        ),
    }

    scale_gradient = (
        readout.log_token_scale.grad
    )

    output_tensors = (
        output.graph_embedding,
        output.normalized_node_features,
        output.graph_context,
        output.attention_logits,
        output.node_gates,
        output.node_weights,
        output.weighted_node_feature,
        output.residual_mean_feature,
        output.readout_gate,
        output.projected_feature,
        output.normalized_projected_feature,
        output.graph_token,
        output.token_scale,
    )

    checks = {
        "graph_embedding_shape": (
            tuple(
                output.graph_embedding.shape
            )
            == (1, 256)
        ),
        "graph_context_shape": (
            tuple(
                output.graph_context.shape
            )
            == (1, 128)
        ),
        "node_weight_shape": (
            tuple(
                output.node_weights.shape
            )
            == (node_count,)
        ),
        "graph_token_shape": (
            tuple(
                output.graph_token.shape
            )
            == (1, 1, 3584)
        ),
        "weights_positive": bool(
            (output.node_weights > 0.0).all()
        ),
        "weights_normalized": bool(
            torch.allclose(
                output.node_weights.sum(),
                torch.tensor(1.0),
                atol=1e-6,
                rtol=0.0,
            )
        ),
        "node_gate_range": bool(
            (output.node_gates >= 0.0).all()
            and (output.node_gates <= 1.0).all()
        ),
        "readout_gate_range": bool(
            (output.readout_gate >= 0.0).all()
            and (output.readout_gate <= 1.0).all()
        ),
        "initial_node_gate": (
            0.40
            <= float(
                output.node_gates.mean().item()
            )
            <= 0.60
        ),
        "initial_readout_gate": (
            0.15
            <= float(
                output.readout_gate.mean().item()
            )
            <= 0.40
        ),
        "token_scale": (
            abs(token_scale - 0.35)
            <= 1e-6
        ),
        "normalized_rms": (
            abs(normalized_rms - 1.0)
            <= 1e-4
        ),
        "token_norm": (
            token_norm_error <= 1e-3
        ),
        "deterministic_eval": (
            repeat_error == 0.0
        ),
        "permutation_token": (
            permutation_token_error <= 1e-6
        ),
        "permutation_weights": (
            permutation_weight_error <= 1e-6
        ),
        "single_node_weight": bool(
            torch.allclose(
                single_output.node_weights,
                torch.ones(1),
                atol=1e-7,
                rtol=0.0,
            )
        ),
        "prior_changes_bat_weight": (
            focused_bat_weight
            > baseline_bat_weight
        ),
        "finite_outputs": all(
            bool(torch.isfinite(tensor).all())
            for tensor in output_tensors
        ),
        "finite_loss": bool(
            torch.isfinite(loss)
        ),
        "gnn_output_gradient": bool(
            gnn_output
            .final_node_features
            .grad is not None
            and torch.isfinite(
                gnn_output
                .final_node_features
                .grad
            ).all()
        ),
        "node_context_gradient": bool(
            node_context.grad is not None
            and torch.isfinite(
                node_context.grad
            ).all()
            and node_context.grad
            .abs()
            .sum()
            .item() > 0.0
        ),
        "visual_gradient": (
            visual_gradient_norm > 0.0
        ),
        "profile_gradient": (
            profile_gradient_norm > 0.0
        ),
        "module_gradients": all(
            value > 0.0
            for value
            in module_gradients.values()
        ),
        "scale_gradient": bool(
            scale_gradient is not None
            and torch.isfinite(
                scale_gradient
            ).all()
            and scale_gradient
            .abs()
            .item() > 0.0
        ),
    }

    print(
        "Gated Graph Readout summary:"
    )
    print(
        "  Nodes:",
        node_count,
    )
    print(
        "  GNN features:",
        tuple(
            gnn_output
            .final_node_features
            .shape
        ),
    )
    print(
        "  Graph embedding:",
        tuple(
            output.graph_embedding.shape
        ),
    )
    print(
        "  Graph Token:",
        tuple(
            output.graph_token.shape
        ),
    )
    print(
        "  GNN parameters:",
        f"{parameter_count(gnn):,}",
    )
    print(
        "  Readout parameters:",
        f"{parameter_count(readout):,}",
    )

    print("\nReadout diagnostics:")
    print(
        "  Node gate mean:",
        f"{output.node_gates.mean().item():.8f}",
    )
    print(
        "  Readout gate mean:",
        f"{output.readout_gate.mean().item():.8f}",
    )
    print(
        "  Normalized weight entropy:",
        f"{normalized_entropy:.8f}",
    )
    print(
        "  Prior/weight correlation:",
        f"{prior_weight_correlation:.8f}",
    )
    print(
        "  Baseline correct-bat weight:",
        f"{baseline_bat_weight:.8f}",
    )
    print(
        "  Focused correct-bat weight:",
        f"{focused_bat_weight:.8f}",
    )

    print("\nNode weights:")

    for index, node in enumerate(nodes):
        print(
            f"  N{index:02d} "
            f"class={node.class_name:<16} "
            f"det={node.detection_id:>3} "
            f"prior={node_prior[index].item():.4f} "
            f"gate={output.node_gates[index].item():.4f} "
            f"weight={output.node_weights[index].item():.4f}"
        )

    print("\nGraph Token diagnostics:")
    print(
        "  Normalized RMS:",
        f"{normalized_rms:.8f}",
    )
    print(
        "  Token scale:",
        f"{token_scale:.8f}",
    )
    print(
        "  Token norm:",
        f"{token_norm:.8f}",
    )
    print(
        "  Expected norm:",
        f"{expected_token_norm:.8f}",
    )
    print(
        "  Norm error:",
        f"{token_norm_error:.8f}",
    )
    print(
        "  Repeat error:",
        f"{repeat_error:.10f}",
    )
    print(
        "  Permutation token error:",
        f"{permutation_token_error:.10f}",
    )
    print(
        "  Permutation weight error:",
        f"{permutation_weight_error:.10f}",
    )

    print("\nGradient diagnostics:")
    print(
        "  Test loss:",
        f"{loss.item():.8f}",
    )
    print(
        "  Visual gradient:",
        f"{visual_gradient_norm:.8f}",
    )
    print(
        "  Profile gradient:",
        f"{profile_gradient_norm:.8f}",
    )

    for name, value in (
        module_gradients.items()
    ):
        print(
            f"  {name} gradient:",
            f"{value:.8f}",
        )

    print(
        "  token scale gradient:",
        f"{scale_gradient.item():.8f}",
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
            "Gated Graph Readout test "
            f"failed: {failed}"
        )

    print(
        "\nGated Graph Readout test: PASS"
    )
    print(
        "Note: this validates readout weighting "
        "and Graph Token projection, not learned "
        "task performance."
    )


if __name__ == "__main__":
    main()