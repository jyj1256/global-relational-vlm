import json
import math
import sys
from collections import Counter
from pathlib import Path

import bitsandbytes as bnb
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoModel,
    BitsAndBytesConfig,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

VLM_ROOT = PROJECT_ROOT / "vlm"

for path in (
    PROJECT_ROOT,
    VLM_ROOT,
):
    if str(path) not in sys.path:
        sys.path.insert(
            0,
            str(path),
        )


from scripts.test_backbone import (
    get_vision_device,
    load_image,
)
from vlm.detectors.detection_types import (
    DetectionBox,
)
from vlm.models.graph_node_builder import (
    extract_projected_visual_features,
)
from vlm.models.object_graph_v1.encoder import (
    ObjectGraphTokenEncoderV1,
)


MODEL_NAME = "OpenGVLab/InternVL3-8B"

CACHE_DIR = (
    PROJECT_ROOT
    / "pretrained/huggingface"
)

OFFLOAD_DIR = (
    PROJECT_ROOT
    / "pretrained/offload"
)

PAPER_ROOT = (
    VLM_ROOT
    / "outputs/paper_figures/"
    "baseball_2337002"
)

IMAGE_PATH = PAPER_ROOT / "input.jpg"

DETECTION_JSON = (
    VLM_ROOT
    / "outputs/object_detections/"
    "yolo11s_coco/entries/2337002.json"
)

ATTENTION_PATH = (
    PAPER_ROOT
    / "local_attention_vector.npy"
)

PROFILE_PATH = (
    PAPER_ROOT
    / "local_attention_profiles.pt"
)

METADATA_PATH = (
    PAPER_ROOT
    / "metadata.json"
)

OUTPUT_PATH = (
    VLM_ROOT
    / "outputs/object_graph_visualizations/"
    "baseball_2337002/"
    "real_object_graph_encoder_v1_test.pt"
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


def load_attention_profiles(path):
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(payload, dict):
        raise TypeError(
            "Attention Profile artifact "
            "must be a dictionary."
        )

    profiles = payload["profiles"]
    layer_indices = tuple(
        int(value)
        for value
        in payload["layer_indices"]
    )

    if not isinstance(
        profiles,
        torch.Tensor,
    ):
        raise TypeError(
            "profiles must be a tensor."
        )

    profiles = profiles.float()

    if profiles.ndim == 2:
        layer_count = len(layer_indices)

        if (
            profiles.shape[1]
            % layer_count
            != 0
        ):
            raise ValueError(
                "Flattened profile dimension "
                "cannot be divided by layers."
            )

        head_count = (
            profiles.shape[1]
            // layer_count
        )

        profiles = profiles.reshape(
            profiles.shape[0],
            layer_count,
            head_count,
        )

    if profiles.ndim != 3:
        raise ValueError(
            "Profiles must have shape "
            "(tokens, layers, heads)."
        )

    return payload, profiles


def main():
    required_paths = (
        IMAGE_PATH,
        DETECTION_JSON,
        ATTENTION_PATH,
        PROFILE_PATH,
        METADATA_PATH,
    )

    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    torch.manual_seed(2026)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    metadata = json.loads(
        METADATA_PATH.read_text()
    )

    detection_payload, detections = (
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

    (
        profile_payload,
        attention_profiles,
    ) = load_attention_profiles(
        PROFILE_PATH
    )

    expected_tile_grid = (
        int(metadata["tile_columns"]),
        int(metadata["tile_rows"]),
    )

    expected_image_size = (
        int(metadata["original_width"]),
        int(metadata["original_height"]),
    )

    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=(
                torch.bfloat16
            ),
            llm_int8_skip_modules=[
                "vision_model",
                "mlp1",
            ],
        )
    )

    print(
        "Loading NF4 InternVL3-8B..."
    )

    backbone = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        cache_dir=str(CACHE_DIR),
        torch_dtype=torch.bfloat16,
        quantization_config=(
            quantization_config
        ),
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory={
            0: "10GiB",
            "cpu": "48GiB",
        },
        offload_folder=str(OFFLOAD_DIR),
        use_flash_attn=False,
        attn_implementation="eager",
        local_files_only=True,
    ).eval()

    backbone.requires_grad_(False)

    quantized_module_names = [
        name
        for name, module
        in backbone.named_modules()
        if isinstance(
            module,
            bnb.nn.Linear4bit,
        )
    ]

    outside_language_quantized = [
        name
        for name in quantized_module_names
        if not name.startswith(
            "language_model"
        )
    ]

    (
        original_image,
        pixel_values,
        tile_grid,
    ) = load_image(
        image_path=IMAGE_PATH,
        image_size=448,
        max_num=1,
    )

    vision_device = get_vision_device(
        backbone
    )

    pixel_values = pixel_values.to(
        device=vision_device,
        dtype=torch.bfloat16,
    )

    print(
        "Extracting real projected "
        "visual features..."
    )

    with torch.inference_mode():
        projected_visual_features = (
            extract_projected_visual_features(
                model=backbone,
                pixel_values=pixel_values,
            )
        )

    projected_visual_features = (
        projected_visual_features
        .detach()
        .clone()
        .requires_grad_(True)
    )

    attention_profiles = (
        attention_profiles
        .detach()
        .requires_grad_(True)
    )

    graph_encoder = (
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
            dropout=0.0,
            initial_token_scale=0.35,
        )
        .to(vision_device)
        .eval()
    )

    print(
        "Running real Object Graph "
        "Token Encoder v1..."
    )

    output = graph_encoder(
        detections=detections,
        local_attention=local_attention,
        projected_visual_features=(
            projected_visual_features
        ),
        attention_profiles=(
            attention_profiles
        ),
        tile_grid=tile_grid,
        image_size=original_image.size,
    )

    nodes = output.hybrid_output.nodes
    edges = output.graph_output.edges

    source_counts = Counter(
        node.source
        for node in nodes
    )
    class_counts = Counter(
        node.class_name
        for node in nodes
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
            "Correct bat node was not "
            "retained exactly once."
        )

    correct_bat_node = (
        correct_bat_nodes[0]
    )

    correct_bat_degree = int(
        output.graph_output.degree[
            correct_bat_node.node_id
        ].item()
    )

    correct_bat_weight = float(
        output
        .readout_output
        .node_weights[
            correct_bat_node.node_id
        ]
        .item()
    )

    token_scale = float(
        output
        .readout_output
        .token_scale
        .item()
    )

    graph_token_norm = float(
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
        device=vision_device,
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

    backbone_has_gradient = any(
        parameter.grad is not None
        for parameter in backbone.parameters()
    )

    finite_outputs = all(
        bool(torch.isfinite(tensor).all())
        for tensor in (
            projected_visual_features,
            attention_profiles,
            output.graph_token,
            output
            .gnn_output
            .final_node_features,
            output
            .readout_output
            .node_weights,
        )
    )

    checks = {
        "model_loaded_in_4bit": bool(
            getattr(
                backbone,
                "is_loaded_in_4bit",
                False,
            )
        ),
        "language_quantized": (
            len(
                quantized_module_names
            )
            > 0
        ),
        "nothing_outside_language_quantized": (
            len(
                outside_language_quantized
            )
            == 0
        ),
        "image_size": (
            original_image.size
            == expected_image_size
        ),
        "tile_grid": (
            tuple(tile_grid)
            == expected_tile_grid
        ),
        "pixel_tensor_shape": (
            tuple(pixel_values.shape)
            == (1, 3, 448, 448)
        ),
        "visual_feature_shape": (
            tuple(
                projected_visual_features
                .shape
            )
            == (1, 256, 3584)
        ),
        "attention_shape": (
            tuple(
                local_attention.shape
            )
            == (256,)
        ),
        "attention_sum": (
            abs(
                float(
                    local_attention
                    .sum()
                    .item()
                )
                - 1.0
            )
            <= 1e-6
        ),
        "profile_shape": (
            tuple(
                attention_profiles.shape
            )
            == (256, 8, 28)
        ),
        "profile_layers": (
            tuple(
                int(value)
                for value
                in profile_payload[
                    "layer_indices"
                ]
            )
            == (
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
            )
        ),
        "detection_count": (
            len(detections) == 30
        ),
        "hybrid_node_count": (
            1 <= len(nodes) <= 12
        ),
        "correct_bat_fused": (
            correct_bat_node.source
            == "fused"
        ),
        "correct_bat_connected": (
            correct_bat_degree >= 1
        ),
        "sparse_graph": (
            0
            < len(edges)
            < (
                len(nodes)
                * (len(nodes) - 1)
                // 2
            )
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
                torch.tensor(
                    1.0,
                    device=vision_device,
                ),
                atol=1e-6,
                rtol=0.0,
            )
        ),
        "normalized_rms": (
            abs(normalized_rms - 1.0)
            <= 1e-4
        ),
        "token_norm": (
            abs(
                graph_token_norm
                - expected_token_norm
            )
            <= 1e-3
        ),
        "finite_outputs": finite_outputs,
        "finite_loss": bool(
            torch.isfinite(loss)
        ),
        "visual_input_gradient": bool(
            projected_visual_features.grad
            is not None
            and torch.isfinite(
                projected_visual_features.grad
            ).all()
        ),
        "profile_input_gradient": bool(
            attention_profiles.grad
            is not None
            and torch.isfinite(
                attention_profiles.grad
            ).all()
        ),
        "encoder_gradient": (
            gradient_norm(
                graph_encoder
            )
            > 0.0
        ),
        "backbone_frozen": (
            not backbone_has_gradient
        ),
    }

    print(
        "\nReal Object Graph Encoder summary:"
    )
    print(
        "  Image:",
        IMAGE_PATH,
    )
    print(
        "  Image size:",
        original_image.size,
    )
    print(
        "  Tile grid:",
        tile_grid,
    )
    print(
        "  Pixel tensor:",
        tuple(pixel_values.shape),
    )
    print(
        "  Visual features:",
        tuple(
            projected_visual_features.shape
        ),
        projected_visual_features.dtype,
    )
    print(
        "  Attention profiles:",
        tuple(
            attention_profiles.shape
        ),
    )
    print(
        "  YOLO detections:",
        len(detections),
    )
    print(
        "  Hybrid nodes:",
        len(nodes),
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
        len(edges),
    )
    print(
        "  Trainable parameters:",
        f"{parameter_count(graph_encoder):,}",
    )

    print("\nReal node weights:")

    for node in nodes:
        weight = float(
            output
            .readout_output
            .node_weights[
                node.node_id
            ]
            .item()
        )

        degree = int(
            output.graph_output.degree[
                node.node_id
            ].item()
        )

        print(
            f"  N{node.node_id:02d} "
            f"source={node.source:<9} "
            f"class={node.class_name:<16} "
            f"det={node.detection_id:>3} "
            f"degree={degree} "
            f"prior={node.selection_score:.4f} "
            f"weight={weight:.4f}"
        )

    print("\nStrongest real edges:")

    for edge in sorted(
        edges,
        key=lambda item: -item.weight,
    ):
        first = nodes[
            edge.first_node
        ]
        second = nodes[
            edge.second_node
        ]

        print(
            f"  N{edge.first_node:02d}"
            f"({first.class_name}) "
            f"<-> "
            f"N{edge.second_node:02d}"
            f"({second.class_name}) "
            f"w={edge.weight:.3f} "
            f"spatial="
            f"{edge.spatial_similarity:.3f} "
            f"visual="
            f"{edge.visual_similarity:.3f} "
            f"attention="
            f"{edge.attention_similarity:.3f} "
            f"IoU={edge.overlap_iou:.3f}"
        )

    print("\nGraph Token diagnostics:")
    print(
        "  Correct bat degree:",
        correct_bat_degree,
    )
    print(
        "  Correct bat weight:",
        f"{correct_bat_weight:.8f}",
    )
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
        f"{graph_token_norm:.8f}",
    )
    print(
        "  Expected norm:",
        f"{expected_token_norm:.8f}",
    )
    print(
        "  Test loss:",
        f"{loss.item():.8f}",
    )
    print(
        "  Encoder gradient:",
        f"{gradient_norm(graph_encoder):.8f}",
    )

    if torch.cuda.is_available():
        print(
            "  Peak CUDA memory:",
            f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
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
            "Real Object Graph Encoder test "
            f"failed: {failed}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "graph_token": (
                output.graph_token
                .detach()
                .cpu()
            ),
            "node_weights": (
                output
                .readout_output
                .node_weights
                .detach()
                .cpu()
            ),
            "nodes": output.hybrid_output.nodes,
            "edges": output.graph_output.edges,
            "class_indices": (
                output.class_indices
                .detach()
                .cpu()
            ),
            "visual_feature_shape": tuple(
                projected_visual_features
                .shape
            ),
            "attention_profile_shape": (
                tuple(
                    attention_profiles.shape
                )
            ),
            "trained": False,
            "features": "real_internvl_nf4",
        },
        OUTPUT_PATH,
    )

    print(
        "\nSaved real-feature artifact:",
        OUTPUT_PATH,
    )

    print(
        "\nReal Object Graph Token "
        "Encoder v1 test: PASS"
    )
    print(
        "Note: the InternVL features are real, "
        "but the Object Graph Encoder remains "
        "untrained."
    )


if __name__ == "__main__":
    main()