from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.offsetbox import (
    AnnotationBbox,
    OffsetImage,
)
from matplotlib.patches import (
    Patch,
    Rectangle,
)
from PIL import Image


SOURCE_COLORS = {
    "fused": "#E63946",
    "detection": "#168AAD",
    "attention": "#F4A261",
}

SOURCE_LABELS = {
    "fused": "YOLO + Attention",
    "detection": "YOLO detection",
    "attention": "Attention residual",
}

CLASS_ALIASES = {
    "baseball bat": "bat",
    "baseball glove": "glove",
    "__attention__": "attention",
}


def _as_path(path):
    return Path(path).expanduser().resolve()


def _load_rgb_image(path):
    path = _as_path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    return Image.open(path).convert("RGB")


def _node_label(node):
    class_name = CLASS_ALIASES.get(
        node.class_name,
        node.class_name,
    )

    return f"N{node.node_id:02d} {class_name}"


def _node_color(node):
    return SOURCE_COLORS.get(
        node.source,
        "#6C757D",
    )


def _normalized_weights(
    node_weights,
    node_count,
):
    weights = np.asarray(
        node_weights,
        dtype=np.float32,
    ).reshape(-1)

    if weights.shape[0] != node_count:
        raise ValueError(
            "node_weights does not match "
            "the node count."
        )

    if not np.isfinite(weights).all():
        raise ValueError(
            "node_weights contains NaN or Inf."
        )

    if (weights < 0.0).any():
        raise ValueError(
            "node_weights must be non-negative."
        )

    total = float(weights.sum())

    if total <= 0.0:
        raise ValueError(
            "node_weights must have a "
            "positive sum."
        )

    return weights / total


def _crop_node_image(
    image,
    normalized_xyxy,
    padding_ratio=0.10,
    canvas_size=160,
):
    image_width, image_height = image.size
    x1, y1, x2, y2 = normalized_xyxy

    left = float(x1) * image_width
    top = float(y1) * image_height
    right = float(x2) * image_width
    bottom = float(y2) * image_height

    box_width = max(
        right - left,
        2.0,
    )
    box_height = max(
        bottom - top,
        2.0,
    )

    padding_x = (
        box_width * padding_ratio + 2.0
    )
    padding_y = (
        box_height * padding_ratio + 2.0
    )

    left = max(
        0,
        int(np.floor(left - padding_x)),
    )
    top = max(
        0,
        int(np.floor(top - padding_y)),
    )
    right = min(
        image_width,
        int(np.ceil(right + padding_x)),
    )
    bottom = min(
        image_height,
        int(np.ceil(bottom + padding_y)),
    )

    crop = image.crop(
        (left, top, right, bottom)
    )

    crop.thumbnail(
        (canvas_size, canvas_size),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new(
        "RGB",
        (canvas_size, canvas_size),
        color=(245, 245, 245),
    )

    offset_x = (
        canvas_size - crop.width
    ) // 2
    offset_y = (
        canvas_size - crop.height
    ) // 2

    canvas.paste(
        crop,
        (offset_x, offset_y),
    )

    return canvas


def render_hybrid_node_overlay(
    image_path,
    nodes,
    node_weights,
    output_path,
    title=(
        "Object–Attention Hybrid Nodes"
    ),
    dpi=300,
):
    image = _load_rgb_image(
        image_path
    )
    nodes = tuple(nodes)

    if not nodes:
        raise ValueError(
            "At least one node is required."
        )

    weights = _normalized_weights(
        node_weights,
        len(nodes),
    )

    output_path = _as_path(
        output_path
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_width, image_height = image.size

    figure_width = 12.0
    figure_height = (
        figure_width
        * image_height
        / image_width
    )

    figure, axis = plt.subplots(
        figsize=(
            figure_width,
            figure_height,
        )
    )

    axis.imshow(image)

    for node in nodes:
        x1, y1, x2, y2 = (
            node.normalized_xyxy
        )

        left = x1 * image_width
        top = y1 * image_height
        width = (
            x2 - x1
        ) * image_width
        height = (
            y2 - y1
        ) * image_height

        color = _node_color(node)
        weight = float(
            weights[node.node_id]
        )

        line_width = (
            1.5 + 8.0 * weight
        )

        rectangle = Rectangle(
            (left, top),
            width,
            height,
            fill=False,
            edgecolor=color,
            linewidth=line_width,
            alpha=0.95,
        )

        axis.add_patch(rectangle)

        label = (
            f"{_node_label(node)} "
            f"| {node.source[0].upper()} "
            f"| w={weight:.3f}"
        )

        axis.text(
            left,
            max(top - 3.0, 2.0),
            label,
            fontsize=7,
            color="white",
            ha="left",
            va="bottom",
            bbox={
                "facecolor": color,
                "edgecolor": "none",
                "alpha": 0.90,
                "pad": 1.5,
            },
        )

    legend_items = [
        Patch(
            facecolor=color,
            edgecolor=color,
            label=SOURCE_LABELS[source],
        )
        for source, color
        in SOURCE_COLORS.items()
    ]

    axis.legend(
        handles=legend_items,
        loc="lower right",
        fontsize=8,
        framealpha=0.90,
    )

    axis.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        pad=10,
    )
    axis.set_xlim(
        0,
        image_width,
    )
    axis.set_ylim(
        image_height,
        0,
    )
    axis.axis("off")

    figure.tight_layout(
        pad=0.4
    )
    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    pdf_path = output_path.with_suffix(
        ".pdf"
    )

    figure.savefig(
        pdf_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(figure)

    return {
        "png": output_path,
        "pdf": pdf_path,
    }


def _graph_layout(
    graph,
    nodes,
    seed=2026,
):
    initial_positions = {
        node.node_id: np.asarray(
            [
                node.normalized_center[0],
                1.0
                - node.normalized_center[1],
            ],
            dtype=np.float64,
        )
        for node in nodes
    }

    if len(nodes) == 1:
        return {
            nodes[0].node_id: np.asarray(
                [0.5, 0.5],
                dtype=np.float64,
            )
        }

    positions = nx.spring_layout(
        graph,
        pos=initial_positions,
        seed=seed,
        k=0.65,
        iterations=200,
        weight="weight",
        scale=1.0,
    )

    x_values = np.asarray(
        [
            positions[
                node.node_id
            ][0]
            for node in nodes
        ]
    )
    y_values = np.asarray(
        [
            positions[
                node.node_id
            ][1]
            for node in nodes
        ]
    )

    x_min = float(x_values.min())
    x_max = float(x_values.max())
    y_min = float(y_values.min())
    y_max = float(y_values.max())

    x_range = max(
        x_max - x_min,
        1e-6,
    )
    y_range = max(
        y_max - y_min,
        1e-6,
    )

    normalized_positions = {}

    for node in nodes:
        x_value, y_value = positions[
            node.node_id
        ]

        normalized_positions[
            node.node_id
        ] = np.asarray(
            [
                0.10
                + 0.80
                * (
                    (x_value - x_min)
                    / x_range
                ),
                0.12
                + 0.76
                * (
                    (y_value - y_min)
                    / y_range
                ),
            ],
            dtype=np.float64,
        )

    return normalized_positions


def render_sparse_relation_graph(
    image_path,
    nodes,
    edges,
    node_weights,
    output_path,
    maximum_edges=None,
    edge_label_count=6,
    title=(
        "Object–Attention Sparse "
        "Relation Graph"
    ),
    seed=2026,
    dpi=300,
):
    image = _load_rgb_image(
        image_path
    )
    nodes = tuple(nodes)
    edges = tuple(edges)

    if not nodes:
        raise ValueError(
            "At least one node is required."
        )

    weights = _normalized_weights(
        node_weights,
        len(nodes),
    )

    ranked_edges = sorted(
        edges,
        key=lambda edge: (
            -float(edge.weight),
            int(edge.edge_id),
        ),
    )

    if maximum_edges is not None:
        if maximum_edges <= 0:
            raise ValueError(
                "maximum_edges must be positive."
            )

        ranked_edges = ranked_edges[
            :maximum_edges
        ]

    graph = nx.Graph()

    for node in nodes:
        graph.add_node(node.node_id)

    for edge in ranked_edges:
        graph.add_edge(
            edge.first_node,
            edge.second_node,
            weight=float(edge.weight),
        )

    positions = _graph_layout(
        graph=graph,
        nodes=nodes,
        seed=seed,
    )

    output_path = _as_path(
        output_path
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(13, 10)
    )

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.axis("off")

    edge_weights = np.asarray(
        [
            float(edge.weight)
            for edge in ranked_edges
        ],
        dtype=np.float32,
    )

    if len(edge_weights) > 0:
        minimum_weight = float(
            edge_weights.min()
        )
        maximum_weight = float(
            edge_weights.max()
        )
    else:
        minimum_weight = 0.0
        maximum_weight = 1.0

    weight_range = max(
        maximum_weight - minimum_weight,
        1e-6,
    )

    color_map = plt.get_cmap(
        "viridis"
    )

    for edge_rank, edge in enumerate(
        ranked_edges
    ):
        first_position = positions[
            edge.first_node
        ]
        second_position = positions[
            edge.second_node
        ]

        normalized_weight = (
            (
                float(edge.weight)
                - minimum_weight
            )
            / weight_range
        )

        line_width = (
            0.8
            + 4.5 * normalized_weight
        )
        line_alpha = (
            0.35
            + 0.60 * normalized_weight
        )

        axis.plot(
            [
                first_position[0],
                second_position[0],
            ],
            [
                first_position[1],
                second_position[1],
            ],
            color=color_map(
                normalized_weight
            ),
            linewidth=line_width,
            alpha=line_alpha,
            zorder=1,
        )

        if edge_rank < edge_label_count:
            midpoint = (
                first_position
                + second_position
            ) / 2.0

            axis.text(
                midpoint[0],
                midpoint[1],
                f"{edge.weight:.2f}",
                fontsize=7,
                color="#212529",
                ha="center",
                va="center",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "#ADB5BD",
                    "alpha": 0.85,
                    "pad": 1.0,
                },
                zorder=2,
            )

    for node in nodes:
        position = positions[
            node.node_id
        ]
        node_weight = float(
            weights[node.node_id]
        )

        crop = _crop_node_image(
            image=image,
            normalized_xyxy=(
                node.normalized_xyxy
            ),
            padding_ratio=0.12,
            canvas_size=160,
        )

        zoom = (
            0.25
            + 0.40 * node_weight
        )

        image_box = OffsetImage(
            np.asarray(crop),
            zoom=zoom,
        )

        annotation = AnnotationBbox(
            image_box,
            position,
            frameon=True,
            pad=0.08,
            bboxprops={
                "edgecolor": (
                    _node_color(node)
                ),
                "linewidth": (
                    2.0
                    + 6.0 * node_weight
                ),
                "facecolor": "white",
            },
            zorder=3,
        )

        axis.add_artist(annotation)

        axis.text(
            position[0],
            position[1] - 0.075,
            (
                f"{_node_label(node)}\n"
                f"w={node_weight:.3f}"
            ),
            fontsize=8,
            fontweight="bold",
            color="#212529",
            ha="center",
            va="top",
            zorder=4,
        )

    legend_items = [
        Patch(
            facecolor="white",
            edgecolor=color,
            linewidth=2.5,
            label=SOURCE_LABELS[source],
        )
        for source, color
        in SOURCE_COLORS.items()
    ]

    axis.legend(
        handles=legend_items,
        loc="lower left",
        fontsize=9,
        framealpha=0.95,
    )

    axis.set_title(
        title,
        fontsize=15,
        fontweight="bold",
        pad=14,
    )

    axis.text(
        0.5,
        0.015,
        (
            "Node borders indicate source; "
            "edge width/color indicate "
            "relation strength. "
            "Weights are untrained."
        ),
        fontsize=8,
        color="#495057",
        ha="center",
        va="bottom",
    )

    figure.tight_layout(
        pad=0.5
    )
    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    pdf_path = output_path.with_suffix(
        ".pdf"
    )

    figure.savefig(
        pdf_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(figure)

    return {
        "png": output_path,
        "pdf": pdf_path,
    }


def compose_object_graph_figure(
    input_image_path,
    attention_overlay_path,
    hybrid_overlay_path,
    graph_image_path,
    output_path,
    question=None,
    answer=None,
    dpi=300,
):
    input_image = _load_rgb_image(
        input_image_path
    )
    attention_image = _load_rgb_image(
        attention_overlay_path
    )
    hybrid_image = _load_rgb_image(
        hybrid_overlay_path
    )
    graph_image = _load_rgb_image(
        graph_image_path
    )

    output_path = _as_path(
        output_path
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(16, 12),
    )

    panels = (
        (
            axes[0, 0],
            input_image,
            "(a) Input image",
        ),
        (
            axes[0, 1],
            attention_image,
            "(b) Question-conditioned Attention",
        ),
        (
            axes[1, 0],
            hybrid_image,
            "(c) Object–Attention Hybrid Nodes",
        ),
        (
            axes[1, 1],
            graph_image,
            "(d) Sparse Object Relation Graph",
        ),
    )

    for axis, image, title in panels:
        axis.imshow(image)
        axis.set_title(
            title,
            fontsize=13,
            fontweight="bold",
            pad=8,
        )
        axis.axis("off")

    subtitle_parts = []

    if question:
        subtitle_parts.append(
            f"Question: {question}"
        )

    if answer:
        subtitle_parts.append(
            f"Answer: {answer}"
        )

    if subtitle_parts:
        figure.suptitle(
            "   |   ".join(
                subtitle_parts
            ),
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )

    figure.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.97)
    )

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    pdf_path = output_path.with_suffix(
        ".pdf"
    )

    figure.savefig(
        pdf_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(figure)

    return {
        "png": output_path,
        "pdf": pdf_path,
    }