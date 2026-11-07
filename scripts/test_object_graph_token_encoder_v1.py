import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
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


from vlm.detectors.detection_types import (
    DetectionBox,
)
from vlm.models.object_graph_v1.encoder import (
    ObjectGraphTokenEncoderV1,
)


DETECTION_JSON = (
    PROJECT_ROOT
    / "vlm/outputs/object_detections/"
    "yolo11s_coco/entries/2337002.json"
)

ATTENTION_PATH = (
    PROJECT_ROOT
    / "vlm/outputs/paper_figures/"
    "baseball_2337002/"
    "local_attention_vector.npy"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "vlm/outputs/"
    "object_graph_visualizations/"
    "baseball_2337002/"
    "untrained_object_graph_encoder_v1_test.pt"
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


def load_detections(path):
    payload = json.loads(
        path.read_text()
    )

    detections = tuple(
        DetectionBox(
            detection_id=int(
                record["detection_id"]
            ),
            class_id=int(
                record["class_id"]
            ),
            class_name=str(
                record["class_name"]
            ),
            confidence=float(
                record["confidence"]
            ),
            xyxy=tuple(
                float(value)
                for value
                in record["xyxy"]
            ),
            normalized_xyxy=tuple(
                float(value)
                for value
                in record[
                    "normalized_xyxy"
                ]
            ),
        )
        for record
        in payload["detections"]
    )

    return payload, detections


def main():
    if not DETECTION_JSON.is_file():
        raise FileNotFoundError(
            DETECTION_JSON
        )

    if not ATTENTION_PATH.is_file():
        raise FileNotFoundError(
            ATTENTION_PATH
        )

    torch.manual_seed(2026)

    payload, detections = (
        load_detections(
            DETECTION_JSON
        )
    )

    local_attention = torch.from_numpy(
        np.load(ATTENTION_PATH)
    ).float()

    local_attention = (
        local_attention
        / local_attention.sum().clamp_min(
            1e-12
        )
    )

    visual_dimension = 32
    token_count = 256
    profile_shape = (8, 28)
    tile_grid = (1, 1)
    image_size = tuple(
        int(value)
        for value
        in payload["image_size"]
    )

    projected_visual_features = (
        torch.randn(
            1,
            token_count,
            visual_dimension,
            dtype=torch.float32,
        )
        .requires_grad_(True)
    )

    attention_profiles = (
        torch.randn(
            token_count,
            *profile_shape,
            dtype=torch.float32,
        )
        .requires_grad_(True)
    )

    model = ObjectGraphTokenEncoderV1(
        visual_dimension=visual_dimension,
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
        dropout=0.0,
        initial_token_scale=0.35,
    )

    model.eval()

    output = model(
        detections=detections,
        local_attention=local_attention,
        projected_visual_features=(
            projected_visual_features
        ),
        attention_profiles=(
            attention_profiles
        ),
        tile_grid=tile_grid,
        image_size=image_size,
    )

    repeated_output = model(
        detections=detections,
        local_attention=local_attention,
        projected_visual_features=(
            projected_visual_features
        ),
        attention_profiles=(
            attention_profiles
        ),
        tile_grid=tile_grid,
        image_size=image_size,
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

    # Verify the fallback path where YOLO finds
    # no objects. Attention-only nodes must still
    # produce a valid Graph Token.
    attention_only_output = model(
        detections=tuple(),
        local_attention=local_attention,
        projected_visual_features=(
            projected_visual_features
        ),
        attention_profiles=(
            attention_profiles
        ),
        tile_grid=tile_grid,
        image_size=image_size,
    )

    nodes = output.hybrid_output.nodes
    attention_only_nodes = (
        attention_only_output
        .hybrid_output
        .nodes
    )

    source_counts = Counter(
        node.source
        for node in nodes
    )
    class_counts = Counter(
        node.class_name
        for node in nodes
        if node.class_id >= 0
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

    if len(correct_bat_nodes) != 1:
        raise RuntimeError(
            "Correct baseball bat node was "
            "not retained exactly once."
        )

    correct_bat_node = (
        correct_bat_nodes[0]
    )

    correct_bat_class_index = int(
        output.class_indices[
            correct_bat_node.node_id
        ].item()
    )

    token_scale = float(
        output
        .readout_output
        .token_scale
        .item()
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

    normalized_rms = float(
        torch.sqrt(
            output
            .readout_output
            .normalized_projected_feature
            .square()
            .mean()
        ).item()
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

    loss = (
        F.mse_loss(
            output.graph_token,
            target,
        )
        + 0.5
        * F.mse_loss(
            attention_only_output
            .graph_token,
            -target,
        )
    )

    loss.backward()

    class_embedding_gradient = (
        model.class_embedding
        .weight
        .grad
    )

    correct_class_gradient = float(
        class_embedding_gradient[
            correct_bat_node.class_id
        ]
        .norm()
        .item()
    )

    unknown_class_gradient = float(
        class_embedding_gradient[
            model.unknown_class_index
        ]
        .norm()
        .item()
    )

    production_model = (
        ObjectGraphTokenEncoderV1(
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
        )
    )

    production_counts = {
        "class_embedding": parameter_count(
            production_model
            .class_embedding
        ),
        "gnn": parameter_count(
            production_model.gnn
        ),
        "readout": parameter_count(
            production_model.readout
        ),
        "total": parameter_count(
            production_model
        ),
    }

    del production_model

    finite_outputs = all(
        bool(torch.isfinite(tensor).all())
        for tensor in (
            output.graph_token,
            output.class_embeddings,
            output.gnn_input_features,
            output
            .gnn_output
            .final_node_features,
            output
            .readout_output
            .node_weights,
            attention_only_output.graph_token,
        )
    )

    main_node_count = len(nodes)
    attention_only_node_count = len(
        attention_only_nodes
    )

    checks = {
        "input_detection_count": (
            len(detections) == 30
        ),
        "proposal_count": (
            1
            <= len(
                output
                .proposal_output
                .regions
            )
            <= 4
        ),
        "hybrid_node_count": (
            1 <= main_node_count <= 12
        ),
        "person_retained": (
            class_counts["person"] >= 1
        ),
        "baseball_bat_retained": (
            class_counts[
                "baseball bat"
            ] >= 1
        ),
        "correct_bat_fused": (
            correct_bat_node.source
            == "fused"
        ),
        "graph_nonempty": (
            output.graph_output
            .undirected_edge_count > 0
        ),
        "graph_sparse": (
            output.graph_output
            .undirected_edge_count
            < (
                main_node_count
                * (main_node_count - 1)
                // 2
            )
        ),
        "class_index_shape": (
            tuple(
                output.class_indices.shape
            )
            == (main_node_count,)
        ),
        "class_embedding_shape": (
            tuple(
                output
                .class_embeddings
                .shape
            )
            == (main_node_count, 32)
        ),
        "correct_bat_class_index": (
            correct_bat_class_index
            == correct_bat_node.class_id
        ),
        "gnn_input_shape": (
            tuple(
                output
                .gnn_input_features
                .shape
            )
            == (
                main_node_count,
                visual_dimension
                + 6
                + 9
                + 32,
            )
        ),
        "gnn_output_shape": (
            tuple(
                output
                .gnn_output
                .final_node_features
                .shape
            )
            == (main_node_count, 256)
        ),
        "graph_token_shape": (
            tuple(
                output.graph_token.shape
            )
            == (1, 1, 3584)
        ),
        "node_weights_normalized": bool(
            torch.allclose(
                output
                .readout_output
                .node_weights
                .sum(),
                torch.tensor(1.0),
                atol=1e-6,
                rtol=0.0,
            )
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
            abs(
                token_norm
                - expected_token_norm
            )
            <= 1e-3
        ),
        "deterministic_eval": (
            repeat_error == 0.0
        ),
        "attention_only_nodes": (
            1
            <= attention_only_node_count
            <= 2
        ),
        "attention_only_sources": all(
            node.source == "attention"
            and node.class_id == -1
            for node
            in attention_only_nodes
        ),
        "unknown_class_indices": bool(
            (
                attention_only_output
                .class_indices
                == model.unknown_class_index
            ).all()
        ),
        "attention_only_token_shape": (
            tuple(
                attention_only_output
                .graph_token
                .shape
            )
            == (1, 1, 3584)
        ),
        "finite_outputs": finite_outputs,
        "finite_loss": bool(
            torch.isfinite(loss)
        ),
        "visual_gradient": bool(
            projected_visual_features
            .grad is not None
            and torch.isfinite(
                projected_visual_features
                .grad
            ).all()
            and projected_visual_features
            .grad
            .abs()
            .sum()
            .item() > 0.0
        ),
        "profile_gradient": bool(
            attention_profiles.grad
            is not None
            and torch.isfinite(
                attention_profiles.grad
            ).all()
            and attention_profiles.grad
            .abs()
            .sum()
            .item() > 0.0
        ),
        "class_embedding_gradient": (
            gradient_norm(
                model.class_embedding
            )
            > 0.0
        ),
        "correct_class_gradient": (
            correct_class_gradient > 0.0
        ),
        "unknown_class_gradient": (
            unknown_class_gradient > 0.0
        ),
        "gnn_gradient": (
            gradient_norm(model.gnn) > 0.0
        ),
        "readout_gradient": (
            gradient_norm(
                model.readout
            )
            > 0.0
        ),
    }

    print(
        "Object Graph Token Encoder v1 summary:"
    )
    print(
        "  Input detections:",
        len(detections),
    )
    print(
        "  Attention Regions:",
        len(
            output
            .proposal_output
            .regions
        ),
    )
    print(
        "  Hybrid Nodes:",
        main_node_count,
    )
    print(
        "  Source counts:",
        dict(source_counts),
    )
    print(
        "  Class counts:",
        dict(class_counts),
    )
    print(
        "  Undirected edges:",
        output.graph_output
        .undirected_edge_count,
    )
    print(
        "  Directed edges:",
        output.graph_output
        .directed_edge_count,
    )

    print("\nShape summary:")
    print(
        "  Pooled visual dimension:",
        visual_dimension,
    )
    print(
        "  Class embeddings:",
        tuple(
            output.class_embeddings.shape
        ),
    )
    print(
        "  GNN input:",
        tuple(
            output
            .gnn_input_features
            .shape
        ),
    )
    print(
        "  GNN output:",
        tuple(
            output
            .gnn_output
            .final_node_features
            .shape
        ),
    )
    print(
        "  Graph Token:",
        tuple(
            output.graph_token.shape
        ),
    )

    print("\nRetained nodes:")

    for node in nodes:
        node_weight = float(
            output
            .readout_output
            .node_weights[
                node.node_id
            ]
            .item()
        )

        print(
            f"  N{node.node_id:02d} "
            f"source={node.source:<9} "
            f"class={node.class_name:<16} "
            f"det={node.detection_id:>3} "
            f"score={node.selection_score:.4f} "
            f"weight={node_weight:.4f}"
        )

    print("\nAttention-only fallback:")
    print(
        "  Nodes:",
        attention_only_node_count,
    )
    print(
        "  Class indices:",
        attention_only_output
        .class_indices
        .tolist(),
    )
    print(
        "  Graph Token:",
        tuple(
            attention_only_output
            .graph_token
            .shape
        ),
    )

    print("\nGraph Token diagnostics:")
    print(
        "  Token scale:",
        f"{token_scale:.8f}",
    )
    print(
        "  Normalized RMS:",
        f"{normalized_rms:.8f}",
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
        "  Repeat error:",
        f"{repeat_error:.10f}",
    )

    print("\nProduction parameter counts:")
    for name, count in (
        production_counts.items()
    ):
        print(
            f"  {name}: {count:,}"
        )

    print("\nGradient diagnostics:")
    print(
        "  Test loss:",
        f"{loss.item():.8f}",
    )
    print(
        "  Visual gradient:",
        f"{projected_visual_features.grad.norm().item():.8f}",
    )
    print(
        "  Profile gradient:",
        f"{attention_profiles.grad.norm().item():.8f}",
    )
    print(
        "  Class embedding gradient:",
        f"{gradient_norm(model.class_embedding):.8f}",
    )
    print(
        "  Correct-class gradient:",
        f"{correct_class_gradient:.8f}",
    )
    print(
        "  Unknown-class gradient:",
        f"{unknown_class_gradient:.8f}",
    )
    print(
        "  GNN gradient:",
        f"{gradient_norm(model.gnn):.8f}",
    )
    print(
        "  Readout gradient:",
        f"{gradient_norm(model.readout):.8f}",
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
            "Object Graph Token Encoder v1 "
            f"test failed: {failed}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": (
                model.state_dict()
            ),
            "graph_token": (
                output.graph_token
                .detach()
                .cpu()
            ),
            "class_indices": (
                output.class_indices
                .detach()
                .cpu()
            ),
            "selected_detection_ids": (
                output.hybrid_output
                .selected_detection_ids
            ),
            "retained_attention_region_ids": (
                output.hybrid_output
                .retained_attention_region_ids
            ),
            "production_parameter_counts": (
                production_counts
            ),
            "trained": False,
        },
        OUTPUT_PATH,
    )

    print(
        "\nSaved untrained test artifact:",
        OUTPUT_PATH,
    )

    print(
        "\nObject Graph Token Encoder v1 "
        "test: PASS"
    )
    print(
        "Note: this validates the integrated "
        "encoder with synthetic visual features, "
        "not task performance."
    )


if __name__ == "__main__":
    main()