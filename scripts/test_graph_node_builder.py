import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from vlm.models.graph_node_builder import (
    build_graph_nodes,
    extract_projected_visual_features,
)
from vlm.models.salient_region_selector import (
    select_salient_regions,
)
from vlm.scripts.test_backbone import (
    get_vision_device,
    load_image,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build Graph Nodes from InternVL features."
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
    parser.add_argument("--max-num", type=int, default=4)
    parser.add_argument("--tile-rows", type=int, default=2)
    parser.add_argument("--tile-cols", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/dm-magneto/dm_yunjin/"
            "vlm/outputs/attention_maps/candy/"
            "graph_nodes_top16.pt"
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
    return parser.parse_args()


def load_model(args):
    args.offload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading InternVL3-8B...")

    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir=(
            "/home/dm-magneto/dm_yunjin/"
            "pretrained/huggingface"
        ),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory={
            0: "10GiB",
            "cpu": "48GiB",
        },
        offload_folder=str(args.offload_dir),
        offload_state_dict=True,
        use_flash_attn=False,
        attn_implementation="eager",
    ).eval()

    return model


def feature_diagnostics(features):
    features = features.float()
    feature_norms = torch.linalg.vector_norm(
        features,
        dim=-1,
    )

    normalized = F.normalize(
        features,
        p=2,
        dim=-1,
    )
    cosine = normalized @ normalized.T

    mask = ~torch.eye(
        cosine.shape[0],
        dtype=torch.bool,
        device=cosine.device,
    )
    off_diagonal = cosine[mask]

    return {
        "norm_min": float(feature_norms.min()),
        "norm_mean": float(feature_norms.mean()),
        "norm_max": float(feature_norms.max()),
        "cosine_min": float(off_diagonal.min()),
        "cosine_mean": float(off_diagonal.mean()),
        "cosine_max": float(off_diagonal.max()),
    }


def main():
    args = parse_args()

    loaded_image = load_image(
        str(args.image), 448,
        max_num=args.max_num,
    )

    if torch.is_tensor(loaded_image):
        pixel_values = loaded_image
    elif isinstance(loaded_image, (tuple, list)):
        pixel_values = next(
            (
                item
                for item in loaded_image
                if torch.is_tensor(item)
            ),
            None,
        )

        if pixel_values is None:
            raise TypeError(
                "load_image() returned no tensor."
            )
    else:
        raise TypeError(
            "Unexpected load_image() output type: "
            f"{type(loaded_image)}"
        )

    num_local_tiles = (
        args.tile_rows * args.tile_cols
    )

    expected_image_inputs = {
        num_local_tiles,
        num_local_tiles + 1,
    }

    if pixel_values.shape[0] not in expected_image_inputs:
        raise ValueError(
            f"Expected {expected_image_inputs} image inputs, "
            f"but received {pixel_values.shape[0]}."
        )

    with Image.open(args.image) as source_image:
        image_size = source_image.size

    local_attention = np.load(args.attention)

    selection = select_salient_regions(
        local_attention=local_attention,
        tile_grid=(
            args.tile_rows,
            args.tile_cols,
        ),
        token_grid=(16, 16),
        image_size=image_size,
        top_k=args.top_k,
    )

    model = load_model(args)
    vision_device = get_vision_device(model)

    pixel_values = pixel_values.to(
        device=vision_device,
        dtype=torch.bfloat16,
    )

    projected_visual_features = (
        extract_projected_visual_features(
            model=model,
            pixel_values=pixel_values,
        )
    )

    graph_nodes = build_graph_nodes(
        projected_visual_features=(
            projected_visual_features
        ),
        selection=selection,
        num_local_tiles=num_local_tiles,
    )

    diagnostics = feature_diagnostics(
        graph_nodes.visual_features
    )

    print("\nVisual feature summary:")
    print(
        f"  All projected features: "
        f"{tuple(projected_visual_features.shape)}"
    )
    print(
        f"  Graph node features: "
        f"{tuple(graph_nodes.visual_features.shape)}"
    )
    print(
        f"  Feature dtype: "
        f"{graph_nodes.visual_features.dtype}"
    )
    print(
        f"  Feature device: "
        f"{graph_nodes.visual_features.device}"
    )

    print("\nGraph node checks:")
    print(
        f"  Attention score sum: "
        f"{graph_nodes.attention_scores.sum().item():.8f}"
    )
    print(
        f"  Attention weight sum: "
        f"{graph_nodes.attention_weights.sum().item():.8f}"
    )
    print(
        f"  Position range: "
        f"[{graph_nodes.normalized_positions.min().item():.6f}, "
        f"{graph_nodes.normalized_positions.max().item():.6f}]"
    )
    print(
        f"  NaN: "
        f"{torch.isnan(graph_nodes.visual_features).any().item()}"
    )
    print(
        f"  Inf: "
        f"{torch.isinf(graph_nodes.visual_features).any().item()}"
    )

    print("\nFeature diagnostics:")
    for key, value in diagnostics.items():
        print(f"  {key}: {value:.6f}")

    print("\nFirst five nodes:")
    for index in range(min(5, args.top_k)):
        print(
            f"  Node {index + 1}: "
            f"local_token="
            f"{graph_nodes.local_token_indices[index].item()}, "
            f"tile="
            f"{graph_nodes.tile_indices[index].item()}, "
            f"attention="
            f"{graph_nodes.attention_scores[index].item():.8f}, "
            f"weight="
            f"{graph_nodes.attention_weights[index].item():.6f}, "
            f"position="
            f"{graph_nodes.normalized_positions[index].tolist()}"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "visual_features": (
                graph_nodes.visual_features
                .detach()
                .float()
                .cpu()
            ),
            "attention_scores": (
                graph_nodes.attention_scores
                .detach()
                .cpu()
            ),
            "attention_weights": (
                graph_nodes.attention_weights
                .detach()
                .cpu()
            ),
            "normalized_positions": (
                graph_nodes.normalized_positions
                .detach()
                .cpu()
            ),
            "normalized_boxes": (
                graph_nodes.normalized_boxes
                .detach()
                .cpu()
            ),
            "local_token_indices": (
                graph_nodes.local_token_indices
                .detach()
                .cpu()
            ),
            "tile_indices": (
                graph_nodes.tile_indices
                .detach()
                .cpu()
            ),
            "metadata": {
                "model": args.model,
                "image": str(args.image),
                "top_k": args.top_k,
                "num_local_tiles": num_local_tiles,
                "tokens_per_tile": 256,
                "thumbnail_excluded": True,
            },
        },
        args.output,
    )

    print(f"\nSaved Graph Nodes: {args.output}")
    print("\nGraph Node builder test: PASS")


if __name__ == "__main__":
    main()