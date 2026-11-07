import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib.offsetbox import (
    AnnotationBbox,
    OffsetImage,
)
from matplotlib.patches import Patch
from PIL import Image, ImageOps


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from vlm.scripts.test_backbone import (
    load_image,
)
from vlm.utils.training_checkpoint import (
    load_training_checkpoint,
)
from vlm.utils.training_model_setup import (
    load_training_model,
)


DEFAULT_IMAGE = (
    PROJECT_ROOT
    / "vlm/outputs/paper_figures/"
    "baseball_2337002/input.jpg"
)
DEFAULT_TRAINING_ROOT = (
    PROJECT_ROOT
    / "vlm/outputs/training/"
    "debug100_full_epoch1"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "vlm/outputs/"
    "object_graph_visualizations/"
    "baseball_2337002/"
    "sparse_attention_graph_v0.png"
)
DEFAULT_STATISTICS = (
    PROJECT_ROOT
    / "vlm/configs/"
    "gqa_descriptor_statistics_"
    "nf4_qlora_5000_seed2026.json"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render the trained Graph v0 "
            "Attention-region graph."
        )
    )

    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
    )
    parser.add_argument(
        "--question",
        type=str,
        default=(
            "What is the man to the left "
            "of the people holding?"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            DEFAULT_TRAINING_ROOT
            / "checkpoints/"
            "step_000013.pt"
        ),
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        default=(
            DEFAULT_TRAINING_ROOT
            / "run_config.json"
        ),
    )
    parser.add_argument(
        "--statistics",
        type=Path,
        default=DEFAULT_STATISTICS,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="OpenGVLab/InternVL3-8B",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "pretrained/huggingface"
        ),
    )
    parser.add_argument(
        "--offload-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "pretrained/offload"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    parser.add_argument(
        "--edge-label-count",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    return parser.parse_args()


def expand_box(
    normalized_box,
    minimum_width=0.12,
    minimum_height=0.12,
    expansion=2.5,
):
    x1, y1, x2, y2 = [
        float(value)
        for value in normalized_box
    ]

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    width = max(
        (x2 - x1) * expansion,
        minimum_width,
    )
    height = max(
        (y2 - y1) * expansion,
        minimum_height,
    )

    x1 = max(
        0.0,
        center_x - width / 2.0,
    )
    y1 = max(
        0.0,
        center_y - height / 2.0,
    )
    x2 = min(
        1.0,
        center_x + width / 2.0,
    )
    y2 = min(
        1.0,
        center_y + height / 2.0,
    )

    return x1, y1, x2, y2


def crop_region(
    image,
    normalized_box,
):
    width, height = image.size

    x1, y1, x2, y2 = expand_box(
        normalized_box
    )

    left = max(
        0,
        int(math.floor(x1 * width)),
    )
    top = max(
        0,
        int(math.floor(y1 * height)),
    )
    right = min(
        width,
        int(math.ceil(x2 * width)),
    )
    bottom = min(
        height,
        int(math.ceil(y2 * height)),
    )

    if right <= left:
        right = min(
            width,
            left + 1,
        )

    if bottom <= top:
        bottom = min(
            height,
            top + 1,
        )

    crop = image.crop(
        (left, top, right, bottom)
    ).convert("RGB")

    thumbnail = ImageOps.pad(
        crop,
        (180, 120),
        method=Image.Resampling.LANCZOS,
        color=(255, 255, 255),
        centering=(0.5, 0.5),
    )

    return np.asarray(thumbnail)


def render_graph(
    image,
    graph_output,
    output_path,
    seed,
    edge_label_count,
    dpi,
):
    graph_nodes = (
        graph_output.graph_nodes
    )
    weighted_graph = (
        graph_output.weighted_graph
    )

    boxes = (
        graph_nodes
        .normalized_boxes
        .detach()
        .float()
        .cpu()
    )
    positions_in_image = (
        graph_nodes
        .normalized_positions
        .detach()
        .float()
        .cpu()
    )
    attention_weights = (
        graph_nodes
        .attention_weights
        .detach()
        .float()
        .cpu()
        .flatten()
    )
    attention_scores = (
        graph_nodes
        .attention_scores
        .detach()
        .float()
        .cpu()
        .flatten()
    )
    token_indices = (
        graph_nodes
        .local_token_indices
        .detach()
        .cpu()
        .flatten()
    )
    edge_weights = (
        weighted_graph
        .pruned_edge_weights
        .detach()
        .float()
        .cpu()
    )

    node_count = int(
        boxes.shape[0]
    )

    graph = nx.Graph()

    for node_index in range(
        node_count
    ):
        graph.add_node(node_index)

    retained_edges = []

    for first in range(node_count):
        for second in range(
            first + 1,
            node_count,
        ):
            weight = float(
                edge_weights[
                    first,
                    second,
                ].item()
            )

            if weight <= 0.0:
                continue

            graph.add_edge(
                first,
                second,
                weight=weight,
            )
            retained_edges.append(
                (first, second, weight)
            )

    if node_count == 1:
        layout = {
            0: np.asarray(
                [0.0, 0.0]
            )
        }
    else:
        layout = nx.kamada_kawai_layout(
            graph,
            weight=None,
            scale=1.0,
        )

        pair_distances = []

        for first in range(node_count):
            for second in range(
                first + 1,
                node_count,
            ):
                distance = float(
                    np.linalg.norm(
                        layout[first]
                        - layout[second]
                    )
                )
                pair_distances.append(
                    distance
                )

        minimum_distance = min(
            pair_distances
        )

        if minimum_distance < 0.24:
            layout = nx.circular_layout(
                list(range(node_count)),
                scale=1.0,
            )

    figure, axis = plt.subplots(
        figsize=(18, 14),
    )

    axis.set_title(
        "Graph v0 — Attention-Only "
        "Weighted Relation Graph",
        fontsize=24,
        fontweight="bold",
        pad=28,
    )
    axis.set_axis_off()
    axis.set_xlim(-1.35, 1.35)
    axis.set_ylim(-1.32, 1.30)

    if retained_edges:
        edge_values = np.asarray(
            [
                weight
                for _, _, weight
                in retained_edges
            ],
            dtype=np.float32,
        )

        minimum = float(
            edge_values.min()
        )
        maximum = float(
            edge_values.max()
        )
        denominator = max(
            maximum - minimum,
            1e-12,
        )

        edge_colors = []
        edge_widths = []

        for weight in edge_values:
            normalized = (
                float(weight) - minimum
            ) / denominator

            edge_colors.append(
                plt.cm.viridis(
                    0.15
                    + 0.75 * normalized
                )
            )
            edge_widths.append(
                1.0
                + 5.0 * normalized
            )

        nx.draw_networkx_edges(
            graph,
            pos=layout,
            ax=axis,
            edgelist=[
                (first, second)
                for first, second, _
                in retained_edges
            ],
            width=edge_widths,
            edge_color=edge_colors,
            alpha=0.78,
        )

        strongest_edges = sorted(
            retained_edges,
            key=lambda item: item[2],
            reverse=True,
        )[
            :edge_label_count
        ]

        edge_labels = {
            (first, second): (
                f"{weight:.2f}"
            )
            for first, second, weight
            in strongest_edges
        }

        nx.draw_networkx_edge_labels(
            graph,
            pos=layout,
            edge_labels=edge_labels,
            ax=axis,
            font_size=9,
            font_color="#374151",
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "#d1d5db",
                "alpha": 0.90,
            },
        )

    maximum_attention = max(
        float(
            attention_weights.max().item()
        ),
        1e-12,
    )

    top_indices = set(
        torch.topk(
            attention_weights,
            k=min(3, node_count),
        )
        .indices
        .tolist()
    )

    for node_index in range(
        node_count
    ):
        x, y = layout[node_index]

        crop = crop_region(
            image=image,
            normalized_box=(
                boxes[node_index]
                .tolist()
            ),
        )

        attention = float(
            attention_weights[
                node_index
            ].item()
        )
        relative_attention = (
            attention
            / maximum_attention
        )

        if node_index in top_indices:
            border_color = "#ef4444"
        else:
            border_color = "#f59e0b"

        image_box = OffsetImage(
            crop,
            zoom=0.82,
            interpolation="bilinear",
        )

        annotation = AnnotationBbox(
            image_box,
            (x, y),
            frameon=True,
            pad=0.18,
            bboxprops={
                "boxstyle": "square,pad=0.18",
                "edgecolor": border_color,
                "facecolor": "white",
                "linewidth": (
                    1.8
                    + 3.2
                    * relative_attention
                ),
            },
            zorder=5,
        )

        axis.add_artist(annotation)

        label = (
            f"R{node_index:02d}\n"
            f"a={attention:.3f}"
        )

        radius = max(
            math.sqrt(
                float(x) ** 2
                + float(y) ** 2
            ),
            1e-6,
        )

        label_x = (
            float(x)
            + 0.13
            * float(x)
            / radius
        )
        label_y = (
            float(y)
            + 0.13
            * float(y)
            / radius
        )

        axis.text(
            label_x,
            label_y,
            label,
            ha="center",
            va=(
                "bottom"
                if label_y >= 0.0
                else "top"
            ),
            fontsize=10,
            fontweight="bold",
            color="#111827",
            zorder=6,
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
            },
        )

    legend_handles = [
        Patch(
            facecolor="white",
            edgecolor="#ef4444",
            linewidth=3.0,
            label="Top-3 Attention region",
        ),
        Patch(
            facecolor="white",
            edgecolor="#f59e0b",
            linewidth=2.5,
            label="Attention region",
        ),
    ]

    axis.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=12,
        framealpha=0.95,
    )

    figure.text(
        0.5,
        0.025,
        (
            "Graph v0 uses anonymous salient "
            "Attention patches as nodes; "
            "edge width/color indicate the "
            "pruned relation weight. "
            "a denotes normalized Attention "
            "selection weight."
        ),
        ha="center",
        fontsize=11,
        color="#4b5563",
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(figure)

    return {
        "node_count": node_count,
        "edge_count": len(
            retained_edges
        ),
        "attention_weights": (
            attention_weights.tolist()
        ),
        "attention_scores": (
            attention_scores.tolist()
        ),
        "token_indices": (
            token_indices.tolist()
        ),
        "edges": [
            {
                "first": first,
                "second": second,
                "weight": weight,
            }
            for first, second, weight
            in retained_edges
        ],
    }


def main():
    args = parse_args()

    for path in (
        args.image,
        args.checkpoint,
        args.run_config,
        args.statistics,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    training_configuration = (
        json.loads(
            args.run_config.read_text(
                encoding="utf-8"
            )
        )
    )

    graph_version = str(
        training_configuration.get(
            "graph_version",
            "v0",
        )
    )

    if graph_version != "v0":
        raise ValueError(
            "This visualization requires "
            "a Graph v0 checkpoint."
        )

    print("Loading trained Graph v0 model...")

    components = load_training_model(
        model_name=args.model,
        model_cache_dir=(
            args.model_cache_dir
        ),
        offload_dir=args.offload_dir,
        statistics_path=args.statistics,
        gpu_memory="10GiB",
        cpu_memory="48GiB",
        graph_version="v0",
    )

    integrated_model = components[
        "integrated_model"
    ]
    vision_device = components[
        "vision_device"
    ]

    checkpoint_information = (
        load_training_checkpoint(
            input_path=args.checkpoint,
            integrated_model=(
                integrated_model
            ),
            optimizer=None,
            scheduler=None,
            expected_configuration=(
                training_configuration
            ),
            adapter_name=(
                "global_relational"
            ),
            restore_rng=False,
        )
    )

    integrated_model.eval()

    (
        original_image,
        pixel_values,
        tile_grid,
    ) = load_image(
        image_path=args.image,
        image_size=448,
        max_num=1,
    )

    pixel_values = pixel_values.to(
        device=vision_device,
        dtype=torch.bfloat16,
    )

    print(
        "Extracting Adapter-OFF "
        "First Pass..."
    )

    first_pass = (
        integrated_model
        .extract_first_pass(
            pixel_values=pixel_values,
            question=args.question,
            tile_grid=tile_grid,
            image_size=(
                original_image.size
            ),
        )
    )

    print(
        "Building trained Graph v0..."
    )

    with torch.inference_mode():
        (
            _,
            graph_output,
        ) = (
            integrated_model
            .build_auxiliary_tokens(
                first_pass
            )
        )

    diagnostics = render_graph(
        image=original_image,
        graph_output=graph_output,
        output_path=args.output,
        seed=args.seed,
        edge_label_count=(
            args.edge_label_count
        ),
        dpi=args.dpi,
    )

    diagnostics_path = (
        args.output.with_suffix(
            ".json"
        )
    )

    diagnostics.update({
        "image": str(
            args.image.resolve()
        ),
        "question": args.question,
        "checkpoint": str(
            args.checkpoint.resolve()
        ),
        "checkpoint_optimizer_step": (
            checkpoint_information[
                "progress"
            ]["optimizer_step"]
        ),
        "tile_grid": list(
            tile_grid
        ),
        "image_size": list(
            original_image.size
        ),
    })

    diagnostics_path.write_text(
        json.dumps(
            diagnostics,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nGraph v0 visualization:")
    print(
        "  Nodes:",
        diagnostics["node_count"],
    )
    print(
        "  Edges:",
        diagnostics["edge_count"],
    )
    print(
        "  Attention sum:",
        sum(
            diagnostics[
                "attention_weights"
            ]
        ),
    )
    print(
        "  Checkpoint step:",
        diagnostics[
            "checkpoint_optimizer_step"
        ],
    )
    print("  Image:", args.output)
    print(
        "  Diagnostics:",
        diagnostics_path,
    )


if __name__ == "__main__":
    main()
