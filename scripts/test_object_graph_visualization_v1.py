import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from vlm.utils.object_graph_v1_visualization import (
    compose_object_graph_figure,
    render_hybrid_node_overlay,
    render_sparse_relation_graph,
)


PAPER_ROOT = (
    PROJECT_ROOT
    / "vlm/outputs/paper_figures/"
    "baseball_2337002"
)

GRAPH_ROOT = (
    PROJECT_ROOT
    / "vlm/outputs/"
    "object_graph_visualizations/"
    "baseball_2337002"
)

ARTIFACT_PATH = (
    GRAPH_ROOT
    / "real_object_graph_encoder_v1_test.pt"
)

INPUT_IMAGE = (
    PAPER_ROOT / "input.jpg"
)

ATTENTION_OVERLAY = (
    PAPER_ROOT / "attention_overlay.png"
)

HYBRID_OVERLAY = (
    GRAPH_ROOT / "hybrid_nodes_real.png"
)

FULL_GRAPH_IMAGE = (
    GRAPH_ROOT
    / "sparse_object_graph_real.png"
)

PAPER_GRAPH_IMAGE = (
    GRAPH_ROOT
    / "sparse_object_graph_paper_top12.png"
)

PAPER_FIGURE = (
    GRAPH_ROOT
    / "object_graph_v1_paper_figure.png"
)


def image_diagnostics(path):
    image = Image.open(path).convert(
        "RGB"
    )

    array = np.asarray(
        image,
        dtype=np.float32,
    )

    return {
        "size": image.size,
        "mean": float(array.mean()),
        "std": float(array.std()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def output_exists(path):
    return (
        path.is_file()
        and path.stat().st_size > 0
    )


def main():
    required_paths = (
        ARTIFACT_PATH,
        INPUT_IMAGE,
        ATTENTION_OVERLAY,
    )

    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    artifact = torch.load(
        ARTIFACT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    nodes = tuple(
        artifact["nodes"]
    )
    edges = tuple(
        artifact["edges"]
    )

    node_weights = (
        artifact["node_weights"]
        .detach()
        .cpu()
        .float()
        .numpy()
    )

    if len(nodes) == 0:
        raise RuntimeError(
            "No nodes were saved."
        )

    if len(edges) == 0:
        raise RuntimeError(
            "No edges were saved."
        )

    GRAPH_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Rendering Hybrid Node overlay..."
    )

    hybrid_paths = (
        render_hybrid_node_overlay(
            image_path=INPUT_IMAGE,
            nodes=nodes,
            node_weights=node_weights,
            output_path=HYBRID_OVERLAY,
            title=(
                "Object–Attention Hybrid Nodes "
                "(Untrained Readout)"
            ),
            dpi=300,
        )
    )

    print(
        "Rendering full Sparse Graph..."
    )

    full_graph_paths = (
        render_sparse_relation_graph(
            image_path=INPUT_IMAGE,
            nodes=nodes,
            edges=edges,
            node_weights=node_weights,
            output_path=FULL_GRAPH_IMAGE,
            maximum_edges=None,
            edge_label_count=8,
            title=(
                "Object–Attention Sparse "
                "Relation Graph — All Edges"
            ),
            seed=2026,
            dpi=300,
        )
    )

    print(
        "Rendering paper Sparse Graph..."
    )

    paper_graph_paths = (
        render_sparse_relation_graph(
            image_path=INPUT_IMAGE,
            nodes=nodes,
            edges=edges,
            node_weights=node_weights,
            output_path=PAPER_GRAPH_IMAGE,
            maximum_edges=min(
                12,
                len(edges),
            ),
            edge_label_count=6,
            title=(
                "Object–Attention Sparse "
                "Relation Graph"
            ),
            seed=2026,
            dpi=300,
        )
    )

    print(
        "Composing four-panel figure..."
    )

    paper_figure_paths = (
        compose_object_graph_figure(
            input_image_path=INPUT_IMAGE,
            attention_overlay_path=(
                ATTENTION_OVERLAY
            ),
            hybrid_overlay_path=(
                HYBRID_OVERLAY
            ),
            graph_image_path=(
                PAPER_GRAPH_IMAGE
            ),
            output_path=PAPER_FIGURE,
            question=(
                "What is the man to the left "
                "of the people holding?"
            ),
            answer="bat",
            dpi=300,
        )
    )

    output_paths = {
        "hybrid_png": (
            hybrid_paths["png"]
        ),
        "hybrid_pdf": (
            hybrid_paths["pdf"]
        ),
        "full_graph_png": (
            full_graph_paths["png"]
        ),
        "full_graph_pdf": (
            full_graph_paths["pdf"]
        ),
        "paper_graph_png": (
            paper_graph_paths["png"]
        ),
        "paper_graph_pdf": (
            paper_graph_paths["pdf"]
        ),
        "paper_figure_png": (
            paper_figure_paths["png"]
        ),
        "paper_figure_pdf": (
            paper_figure_paths["pdf"]
        ),
    }

    png_diagnostics = {
        name: image_diagnostics(path)
        for name, path
        in output_paths.items()
        if name.endswith("_png")
    }

    correct_bat_nodes = [
        node
        for node in nodes
        if (
            node.class_name
            == "baseball bat"
            and node.detection_id == 4
        )
    ]

    selected_node_ids = {
        node.node_id
        for node in nodes
    }

    edge_node_ids_valid = all(
        edge.first_node
        in selected_node_ids
        and edge.second_node
        in selected_node_ids
        for edge in edges
    )

    checks = {
        "node_count": (
            len(nodes) == 10
        ),
        "edge_count": (
            len(edges) == 19
        ),
        "node_weight_count": (
            len(node_weights) == len(nodes)
        ),
        "node_weights_finite": bool(
            np.isfinite(
                node_weights
            ).all()
        ),
        "node_weights_normalized": (
            abs(
                float(
                    node_weights.sum()
                )
                - 1.0
            )
            <= 1e-5
        ),
        "correct_bat_present": (
            len(correct_bat_nodes) == 1
        ),
        "edge_node_ids_valid": (
            edge_node_ids_valid
        ),
        "all_outputs_exist": all(
            output_exists(path)
            for path
            in output_paths.values()
        ),
        "image_dimensions": all(
            diagnostics["size"][0]
            >= 1000
            and diagnostics["size"][1]
            >= 600
            for diagnostics
            in png_diagnostics.values()
        ),
        "images_nonblank": all(
            diagnostics["std"] > 5.0
            for diagnostics
            in png_diagnostics.values()
        ),
    }

    print(
        "\nObject Graph Visualization summary:"
    )
    print(
        "  Nodes:",
        len(nodes),
    )
    print(
        "  Full edges:",
        len(edges),
    )
    print(
        "  Paper edges:",
        min(12, len(edges)),
    )
    print(
        "  Node weight sum:",
        f"{node_weights.sum():.8f}",
    )

    print("\nGenerated outputs:")

    for name, path in (
        output_paths.items()
    ):
        print(
            f"  {name}: {path}"
        )
        print(
            "    size:",
            f"{path.stat().st_size / 2**20:.3f} MiB",
        )

    print("\nPNG diagnostics:")

    for name, diagnostics in (
        png_diagnostics.items()
    ):
        print(
            f"  {name}: "
            f"pixels={diagnostics['size']}, "
            f"mean={diagnostics['mean']:.2f}, "
            f"std={diagnostics['std']:.2f}"
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
            "Object Graph Visualization "
            f"test failed: {failed}"
        )

    print(
        "\nObject Graph Visualization "
        "test: PASS"
    )
    print(
        "Note: node and edge weights are "
        "untrained and must not be presented "
        "as learned performance results."
    )


if __name__ == "__main__":
    main()