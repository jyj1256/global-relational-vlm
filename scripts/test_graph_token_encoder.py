import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from vlm.models.graph_node_builder import (
    extract_projected_visual_features,
)
from vlm.models.graph_token_encoder import (
    GraphTokenEncoder,
)
from vlm.scripts.test_backbone import (
    get_vision_device,
    load_image,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test the integrated Graph Token Encoder."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default="OpenGVLab/InternVL3-8B",
    )

    parser.add_argument(
        "--image",
        type=Path,
        default=Path(
            "/home/dm-magneto/dm_yunjin/"
            "vlm/tests/assets/candy.jpg"
        ),
    )

    parser.add_argument(
        "--attention",
        type=Path,
        default=Path(
            "/home/dm-magneto/dm_yunjin/"
            "vlm/outputs/attention_maps/candy/"
            "local_attention_vector.npy"
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
            "untrained_graph_encoder_test.pt"
        ),
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            "/home/dm-magneto/dm_yunjin/"
            "pretrained/huggingface"
        ),
    )

    parser.add_argument(
        "--offload-dir",
        type=Path,
        default=Path(
            "/home/dm-magneto/dm_yunjin/"
            "pretrained/offload"
        ),
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=448,
    )

    parser.add_argument(
        "--max-num",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--gpu-memory",
        type=str,
        default="10GiB",
    )

    parser.add_argument(
        "--cpu-memory",
        type=str,
        default="64GiB",
    )

    return parser.parse_args()


def parameter_count(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def gradient_norm(parameters):
    total = 0.0

    for parameter in parameters:
        if parameter.grad is None:
            continue

        gradient = parameter.grad.detach().float()

        total += float(
            gradient.square().sum().item()
        )

    return total ** 0.5


def load_model(args):
    args.cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.offload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading InternVL3-8B...")

    return AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir=str(args.cache_dir),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory={
            0: args.gpu_memory,
            "cpu": args.cpu_memory,
        },
        offload_folder=str(args.offload_dir),
        offload_state_dict=True,
        use_flash_attn=False,
        attn_implementation="eager",
    ).eval()


def main():
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(args.image)

    if not args.attention.exists():
        raise FileNotFoundError(args.attention)

    if not args.profiles.exists():
        raise FileNotFoundError(args.profiles)

    local_attention = np.load(
        args.attention
    )

    profile_data = torch.load(
        args.profiles,
        map_location="cpu",
        weights_only=True,
    )

    attention_profiles = profile_data["profiles"]

    (
        original_image,
        pixel_values,
        tile_grid,
    ) = load_image(
        image_path=str(args.image),
        image_size=args.image_size,
        max_num=args.max_num,
    )

    num_local_tiles = (
        tile_grid[0] * tile_grid[1]
    )

    model = load_model(args)

    vision_device = get_vision_device(model)

    pixel_values = pixel_values.to(
        device=vision_device,
        dtype=torch.bfloat16,
    )

    with torch.no_grad():
        projected_visual_features = (
            extract_projected_visual_features(
                model=model,
                pixel_values=pixel_values,
            )
        )

    torch.manual_seed(42)

    graph_encoder = GraphTokenEncoder(
        top_k=16,
        token_grid=(16, 16),
        sigma_p=0.25,
        threshold=0.70,
        spatial_weight=0.35,
        visual_weight=0.20,
        attention_weight=0.45,
        node_input_dimension=3587,
        graph_dimension=256,
        gnn_layers=2,
        llm_hidden_dimension=3584,
    )

    counts = {
        "gnn": parameter_count(
            graph_encoder.gnn
        ),
        "graph_head": parameter_count(
            graph_encoder.graph_head
        ),
        "total": parameter_count(
            graph_encoder
        ),
    }

    print("\nGraphTokenEncoder parameters:")

    for name, count in counts.items():
        print(f"  {name}: {count:,}")

    graph_encoder.train()

    training_output = graph_encoder(
        local_attention=local_attention,
        attention_profiles=(
            attention_profiles
        ),
        projected_visual_features=(
            projected_visual_features
        ),
        tile_grid=tile_grid,
        image_size=original_image.size,
    )

    target = torch.linspace(
        -0.1,
        0.1,
        3584,
    ).view(1, 1, 3584)

    test_loss = F.mse_loss(
        training_output.graph_token,
        target,
    )

    test_loss.backward()

    gnn_gradient = gradient_norm(
        graph_encoder.gnn.parameters()
    )

    head_gradient = gradient_norm(
        graph_encoder.graph_head.parameters()
    )

    graph_encoder.eval()

    with torch.no_grad():
        output = graph_encoder(
            local_attention=local_attention,
            attention_profiles=(
                attention_profiles
            ),
            projected_visual_features=(
                projected_visual_features
            ),
            tile_grid=tile_grid,
            image_size=original_image.size,
        )

    graph_token = output.graph_token

    edge_row_sums = (
        output.weighted_graph
        .normalized_edge_weights
        .sum(dim=-1)
    )

    has_nan = bool(
        torch.isnan(graph_token).any().item()
    )

    has_inf = bool(
        torch.isinf(graph_token).any().item()
    )

    print("\nShape summary:")
    print(
        "  Projected visual features:",
        tuple(projected_visual_features.shape),
    )
    print(
        "  Local tiles:",
        num_local_tiles,
    )
    print(
        "  Selected regions:",
        len(output.selection.regions),
    )
    print(
        "  Graph node visual features:",
        tuple(
            output.graph_nodes
            .visual_features
            .shape
        ),
    )
    print(
        "  Selected attention profiles:",
        tuple(
            output
            .selected_attention_profiles
            .shape
        ),
    )
    print(
        "  Weighted node features:",
        tuple(
            output.weighted_graph
            .node_features
            .shape
        ),
    )
    print(
        "  Edge matrix:",
        tuple(
            output.weighted_graph
            .normalized_edge_weights
            .shape
        ),
    )
    print(
        "  GNN node features:",
        tuple(
            output.gnn_output
            .final_node_features
            .shape
        ),
    )
    print(
        "  Graph embedding:",
        tuple(
            output.token_output
            .graph_embedding
            .shape
        ),
    )
    print(
        "  Graph token:",
        tuple(graph_token.shape),
    )

    print("\nNumerical diagnostics:")
    print(
        "  Selected attention mass:",
        f"{output.selection.selected_attention_mass:.8f}",
    )
    print(
        "  Edge row-sum range:",
        f"{edge_row_sums.min().item():.8f} - "
        f"{edge_row_sums.max().item():.8f}",
    )
    print(
        "  Graph token norm:",
        f"{graph_token.norm().item():.8f}",
    )
    print(
        "  Test loss:",
        f"{test_loss.item():.8f}",
    )
    print(
        "  GNN gradient norm:",
        f"{gnn_gradient:.8f}",
    )
    print(
        "  Graph head gradient norm:",
        f"{head_gradient:.8f}",
    )
    print("  NaN:", has_nan)
    print("  Inf:", has_inf)

    checks = {
        "gnn_parameters": (
            counts["gnn"] == 991_494
        ),
        "head_parameters": (
            counts["graph_head"] == 918_016
        ),
        "total_parameters": (
            counts["total"] == 1_909_510
        ),
        "selection_count": (
            len(output.selection.regions) == 16
        ),
        "node_visual_shape": (
            output.graph_nodes
            .visual_features
            .shape
            == (16, 3584)
        ),
        "profile_shape": (
            output
            .selected_attention_profiles
            .shape
            == (16, 224)
        ),
        "weighted_node_shape": (
            output.weighted_graph
            .node_features
            .shape
            == (16, 3587)
        ),
        "edge_shape": (
            output.weighted_graph
            .normalized_edge_weights
            .shape
            == (16, 16)
        ),
        "gnn_shape": (
            output.gnn_output
            .final_node_features
            .shape
            == (16, 256)
        ),
        "embedding_shape": (
            output.token_output
            .graph_embedding
            .shape
            == (1, 256)
        ),
        "graph_token_shape": (
            graph_token.shape == (1, 1, 3584)
        ),
        "edge_normalization": bool(
            torch.allclose(
                edge_row_sums,
                torch.ones_like(edge_row_sums),
                atol=1e-5,
            )
        ),
        "finite": not has_nan and not has_inf,
        "gnn_gradient": gnn_gradient > 0.0,
        "head_gradient": head_gradient > 0.0,
    }

    failed_checks = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    print("\nChecks:")

    for name, passed in checks.items():
        print(
            f"  {name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "trained": False,
            "graph_token": graph_token.cpu(),
            "selected_attention_mass": (
                output.selection
                .selected_attention_mass
            ),
            "local_token_indices": (
                output.graph_nodes
                .local_token_indices
                .cpu()
            ),
            "selected_attention_profiles": (
                output
                .selected_attention_profiles
                .cpu()
            ),
            "node_features": (
                output.weighted_graph
                .node_features
                .cpu()
            ),
            "normalized_edge_weights": (
                output.weighted_graph
                .normalized_edge_weights
                .cpu()
            ),
            "gnn_node_features": (
                output.gnn_output
                .final_node_features
                .cpu()
            ),
            "graph_embedding": (
                output.token_output
                .graph_embedding
                .cpu()
            ),
            "parameter_counts": counts,
        },
        args.output,
    )

    print("\nSaved wrapper test artifact:")
    print(f"  {args.output}")

    if failed_checks:
        raise RuntimeError(
            "GraphTokenEncoder test failed: "
            + ", ".join(failed_checks)
        )

    print("\nGraphTokenEncoder test: PASS")
    print(
        "Note: the Graph Token Encoder is untrained."
    )


if __name__ == "__main__":
    main()