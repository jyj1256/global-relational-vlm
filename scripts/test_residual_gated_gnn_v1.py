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


def mean_off_diagonal_cosine(features):
    if features.shape[0] <= 1:
        return 0.0

    normalized = F.normalize(
        features.float(),
        p=2,
        dim=-1,
        eps=1e-12,
    )

    cosine = torch.matmul(
        normalized,
        normalized.transpose(0, 1),
    )

    mask = ~torch.eye(
        features.shape[0],
        dtype=torch.bool,
        device=features.device,
    )

    return float(
        cosine[mask].mean().item()
    )


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
            self_loop_weight=1.0,
        )
    )

    graph = graph_builder(nodes)

    input_dimension = int(
        graph.node_features.shape[-1]
    )

    model = ResidualGatedGNN(
        input_dimension=input_dimension,
        relation_dimension=(
            graph.relation_dimension
        ),
        hidden_dimension=256,
        layer_count=2,
        relation_hidden_dimension=64,
        gate_hidden_dimension=64,
        dropout=0.0,
        initial_residual_gate=-2.0,
        eps=1e-12,
    )

    model.eval()

    output = model(
        node_features=graph.node_features,
        edge_index=graph.edge_index,
        edge_weight=graph.edge_weight,
        edge_features=graph.edge_features,
    )

    node_count = len(nodes)

    projected_cosine = (
        mean_off_diagonal_cosine(
            output.projected_node_features
        )
    )

    layer_cosines = [
        mean_off_diagonal_cosine(
            layer_output.node_features
        )
        for layer_output
        in output.layer_outputs
    ]

    layer_deltas = []
    previous_features = (
        output.projected_node_features
    )

    for layer_output in output.layer_outputs:
        delta = torch.linalg.vector_norm(
            layer_output.node_features
            - previous_features,
            dim=-1,
        ).mean()

        layer_deltas.append(
            float(delta.item())
        )
        previous_features = (
            layer_output.node_features
        )

    message_gate_means = [
        float(
            layer_output
            .message_gates
            .mean()
            .item()
        )
        for layer_output
        in output.layer_outputs
    ]

    residual_gate_means = [
        float(
            layer_output
            .residual_gates
            .mean()
            .item()
        )
        for layer_output
        in output.layer_outputs
    ]

    gate_ranges_valid = all(
        bool(
            (
                layer_output.message_gates
                >= 0.0
            ).all()
            and (
                layer_output.message_gates
                <= 1.0
            ).all()
            and (
                layer_output.residual_gates
                >= 0.0
            ).all()
            and (
                layer_output.residual_gates
                <= 1.0
            ).all()
        )
        for layer_output
        in output.layer_outputs
    )

    # Permutation equivariance:
    # reorder nodes and remap edge indices.
    permutation = torch.randperm(
        node_count
    )
    old_to_new = torch.empty_like(
        permutation
    )
    old_to_new[permutation] = (
        torch.arange(node_count)
    )

    permuted_edge_index = (
        old_to_new[graph.edge_index]
    )

    permuted_output = model(
        node_features=(
            graph.node_features[
                permutation
            ]
        ),
        edge_index=permuted_edge_index,
        edge_weight=graph.edge_weight,
        edge_features=graph.edge_features,
    )

    restored_permuted_output = (
        permuted_output
        .final_node_features[
            old_to_new
        ]
    )

    permutation_error = float(
        (
            restored_permuted_output
            - output.final_node_features
        )
        .abs()
        .max()
        .item()
    )

    empty_edge_index = torch.empty(
        2,
        0,
        dtype=torch.long,
    )
    empty_edge_weight = torch.empty(
        0,
        dtype=torch.float32,
    )
    empty_edge_features = torch.empty(
        0,
        graph.relation_dimension,
        dtype=torch.float32,
    )

    single_output = model(
        node_features=(
            graph.node_features[:1]
        ),
        edge_index=empty_edge_index,
        edge_weight=empty_edge_weight,
        edge_features=empty_edge_features,
    )

    single_identity_error = float(
        (
            single_output.final_node_features
            - single_output
            .projected_node_features
        )
        .abs()
        .max()
        .item()
    )

    target = torch.linspace(
        -0.1,
        0.1,
        256,
        dtype=torch.float32,
    ).unsqueeze(0).expand(
        node_count,
        -1,
    )

    loss = F.mse_loss(
        output.final_node_features,
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

    input_projection_gradient = (
        gradient_norm(
            model.input_projection
        )
    )

    relation_gradient = sum(
        gradient_norm(
            layer.relation_encoder
        )
        for layer in model.layers
    )
    message_gradient = sum(
        gradient_norm(
            layer.message_network
        )
        for layer in model.layers
    )
    message_gate_gradient = sum(
        gradient_norm(
            layer.message_gate
        )
        for layer in model.layers
    )
    update_gradient = sum(
        gradient_norm(
            layer.update_network
        )
        for layer in model.layers
    )
    residual_gate_gradient = sum(
        gradient_norm(
            layer.residual_gate
        )
        for layer in model.layers
    )

    output_tensors = [
        output.projected_node_features,
        output.final_node_features,
    ]

    for layer_output in output.layer_outputs:
        output_tensors.extend(
            [
                layer_output.node_features,
                layer_output
                .aggregated_messages,
                layer_output.edge_messages,
                layer_output.message_gates,
                layer_output.residual_gates,
            ]
        )

    finite_outputs = all(
        bool(torch.isfinite(tensor).all())
        for tensor in output_tensors
    )

    checks = {
        "layer_count": (
            len(output.layer_outputs) == 2
        ),
        "projected_shape": (
            tuple(
                output
                .projected_node_features
                .shape
            )
            == (node_count, 256)
        ),
        "final_shape": (
            tuple(
                output
                .final_node_features
                .shape
            )
            == (node_count, 256)
        ),
        "layer_shapes": all(
            tuple(
                layer_output
                .node_features
                .shape
            )
            == (node_count, 256)
            for layer_output
            in output.layer_outputs
        ),
        "message_gate_shapes": all(
            tuple(
                layer_output
                .message_gates
                .shape
            )
            == (
                graph.directed_edge_count,
            )
            for layer_output
            in output.layer_outputs
        ),
        "residual_gate_shapes": all(
            tuple(
                layer_output
                .residual_gates
                .shape
            )
            == (node_count, 256)
            for layer_output
            in output.layer_outputs
        ),
        "gate_ranges": gate_ranges_valid,
        "initial_residual_control": all(
            0.05 <= gate_mean <= 0.25
            for gate_mean
            in residual_gate_means
        ),
        "nonzero_layer_updates": all(
            delta > 0.0
            for delta in layer_deltas
        ),
        "controlled_layer_updates": all(
            delta < 1.0
            for delta in layer_deltas
        ),
        "finite_outputs": finite_outputs,
        "finite_loss": bool(
            torch.isfinite(loss)
        ),
        "not_collapsed": (
            layer_cosines[-1] < 0.98
        ),
        "permutation_equivariance": (
            permutation_error <= 1e-5
        ),
        "single_node_identity": (
            single_identity_error <= 1e-7
        ),
        "visual_gradient": (
            visual_gradient_norm > 0.0
        ),
        "profile_gradient": (
            profile_gradient_norm > 0.0
        ),
        "input_projection_gradient": (
            input_projection_gradient > 0.0
        ),
        "relation_gradient": (
            relation_gradient > 0.0
        ),
        "message_gradient": (
            message_gradient > 0.0
        ),
        "message_gate_gradient": (
            message_gate_gradient > 0.0
        ),
        "update_gradient": (
            update_gradient > 0.0
        ),
        "residual_gate_gradient": (
            residual_gate_gradient > 0.0
        ),
    }

    print(
        "Residual Gated GNN summary:"
    )
    print(
        "  Input nodes:",
        tuple(
            graph.node_features.shape
        ),
    )
    print(
        "  Directed edges:",
        graph.directed_edge_count,
    )
    print(
        "  Relation features:",
        tuple(
            graph.edge_features.shape
        ),
    )
    print(
        "  Hidden dimension:",
        256,
    )
    print(
        "  GNN layers:",
        len(output.layer_outputs),
    )
    print(
        "  Trainable parameters:",
        f"{parameter_count(model):,}",
    )

    print("\nLayer diagnostics:")
    print(
        "  Projected cosine mean:",
        f"{projected_cosine:.6f}",
    )

    for layer_index, (
        layer_output,
        cosine_mean,
        delta,
        message_gate_mean,
        residual_gate_mean,
    ) in enumerate(
        zip(
            output.layer_outputs,
            layer_cosines,
            layer_deltas,
            message_gate_means,
            residual_gate_means,
        ),
        start=1,
    ):
        print(
            f"  Layer {layer_index}: "
            f"shape="
            f"{tuple(layer_output.node_features.shape)}, "
            f"cosine={cosine_mean:.6f}, "
            f"delta={delta:.6f}, "
            f"message_gate="
            f"{message_gate_mean:.6f}, "
            f"residual_gate="
            f"{residual_gate_mean:.6f}"
        )

    print("\nStructural diagnostics:")
    print(
        "  Permutation max error:",
        f"{permutation_error:.10f}",
    )
    print(
        "  Single-node identity error:",
        f"{single_identity_error:.10f}",
    )

    print("\nGradient diagnostics:")
    print(
        "  Test loss:",
        f"{loss.item():.8f}",
    )
    print(
        "  Visual input gradient:",
        f"{visual_gradient_norm:.8f}",
    )
    print(
        "  Profile input gradient:",
        f"{profile_gradient_norm:.8f}",
    )
    print(
        "  Input projection gradient:",
        f"{input_projection_gradient:.8f}",
    )
    print(
        "  Relation encoder gradient:",
        f"{relation_gradient:.8f}",
    )
    print(
        "  Message network gradient:",
        f"{message_gradient:.8f}",
    )
    print(
        "  Message gate gradient:",
        f"{message_gate_gradient:.8f}",
    )
    print(
        "  Update network gradient:",
        f"{update_gradient:.8f}",
    )
    print(
        "  Residual gate gradient:",
        f"{residual_gate_gradient:.8f}",
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
            "Residual Gated GNN test "
            f"failed: {failed}"
        )

    print(
        "\nResidual Gated GNN test: PASS"
    )
    print(
        "Note: this validates graph message "
        "passing and oversmoothing control, "
        "not learned relation quality."
    )


if __name__ == "__main__":
    main()