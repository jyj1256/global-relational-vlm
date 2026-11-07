import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch


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
from vlm.models.object_graph_v1.attention_region_proposer import (
    AttentionRegionProposer,
)
from vlm.models.object_graph_v1.hybrid_node_builder import (
    HybridNodeBuilder,
)
from vlm.models.object_graph_v1.region_feature_pooler import (
    RegionFeaturePooler,
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


def load_detections(path):
    payload = json.loads(
        path.read_text()
    )

    detections = []

    for record in payload["detections"]:
        detections.append(
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
        )

    return payload, tuple(detections)


def box_iou(first_box, second_box):
    ax1, ay1, ax2, ay2 = first_box
    bx1, by1, bx2, by2 = second_box

    intersection_width = max(
        0.0,
        min(ax2, bx2) - max(ax1, bx1),
    )
    intersection_height = max(
        0.0,
        min(ay2, by2) - max(ay1, by1),
    )
    intersection = (
        intersection_width
        * intersection_height
    )

    first_area = max(
        0.0,
        ax2 - ax1,
    ) * max(
        0.0,
        ay2 - ay1,
    )
    second_area = max(
        0.0,
        bx2 - bx1,
    ) * max(
        0.0,
        by2 - by1,
    )

    union = (
        first_area
        + second_area
        - intersection
    )

    if union <= 0.0:
        return 0.0

    return intersection / union


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

    tile_grid = (1, 1)
    image_size = tuple(
        int(value)
        for value
        in payload["image_size"]
    )

    proposer = AttentionRegionProposer(
        token_grid_size=16,
        peak_relative_threshold=0.15,
        peak_nms_kernel_size=3,
        growth_relative_threshold=0.55,
        max_region_radius=2,
        smoothing_kernel_size=3,
        min_region_patches=1,
        min_region_mass=0.005,
        max_regions=4,
        border_margin=1,
        border_penalty=0.50,
    )

    proposal_output = proposer(
        local_attention=local_attention,
        tile_grid=tile_grid,
        image_size=image_size,
    )

    attention_regions = (
        proposal_output.regions
    )

    if not attention_regions:
        raise RuntimeError(
            "No Attention Regions were proposed."
        )

    visual_dimension = 32
    profile_shape = (8, 28)
    local_token_count = 256

    projected_visual_features = (
        torch.randn(
            1,
            local_token_count,
            visual_dimension,
            dtype=torch.float32,
        )
        .requires_grad_(True)
    )

    attention_profiles = (
        torch.randn(
            local_token_count,
            *profile_shape,
            dtype=torch.float32,
        )
        .requires_grad_(True)
    )

    pooler = RegionFeaturePooler(
        token_grid_size=16,
        eps=1e-12,
    )

    detection_pool_output = pooler(
        normalized_boxes=[
            detection.normalized_xyxy
            for detection in detections
        ],
        local_attention=local_attention,
        projected_visual_features=(
            projected_visual_features
        ),
        attention_profiles=(
            attention_profiles
        ),
        tile_grid=tile_grid,
    )

    region_pool_output = pooler(
        normalized_boxes=[
            region.normalized_xyxy
            for region in attention_regions
        ],
        local_attention=local_attention,
        projected_visual_features=(
            projected_visual_features
        ),
        attention_profiles=(
            attention_profiles
        ),
        tile_grid=tile_grid,
    )

    builder = HybridNodeBuilder(
        min_detection_confidence=0.20,
        attention_rescue_mass=0.01,
        attention_rescue_peak=0.01,
        min_attention_region_mass=0.01,
        match_overlap_threshold=0.30,
        duplicate_iou_threshold=0.70,
        max_detection_nodes=10,
        max_attention_nodes=2,
        max_total_nodes=12,
        max_nodes_per_class=4,
        attention_support_scale=0.03,
        proposal_score_scale=0.01,
        max_attention_visual_blend=0.35,
        max_region_visual_blend=0.20,
        attention_only_visual_blend=0.75,
    )

    output = builder(
        detections=detections,
        detection_features=(
            detection_pool_output
            .region_features
        ),
        attention_regions=(
            attention_regions
        ),
        attention_region_features=(
            region_pool_output
            .region_features
        ),
    )

    detection_nodes = [
        node
        for node in output.nodes
        if node.source != "attention"
    ]
    attention_nodes = [
        node
        for node in output.nodes
        if node.source == "attention"
    ]
    fused_nodes = [
        node
        for node in output.nodes
        if node.source == "fused"
    ]

    selected_classes = Counter(
        node.class_name
        for node in detection_nodes
    )
    source_counts = Counter(
        node.source
        for node in output.nodes
    )

    selected_detection_ids = set(
        output.selected_detection_ids
    )
    suppressed_detection_ids = set(
        output.suppressed_detection_ids
    )
    all_detection_ids = {
        int(detection.detection_id)
        for detection in detections
    }

    retained_region_ids = set(
        output.retained_attention_region_ids
    )
    discarded_region_ids = set(
        output.discarded_attention_region_ids
    )
    consumed_region_ids = {
        int(match.attention_region_id)
        for match in output.matches
    }
    all_region_ids = {
        int(region.region_id)
        for region in attention_regions
    }

    duplicate_free = True

    for first_index, first_node in enumerate(
        detection_nodes
    ):
        for second_node in detection_nodes[
            first_index + 1:
        ]:
            if (
                first_node.class_id
                != second_node.class_id
            ):
                continue

            if box_iou(
                first_node.normalized_xyxy,
                second_node.normalized_xyxy,
            ) >= 0.70 + 1e-6:
                duplicate_free = False

    detection_scores = [
        node.selection_score
        for node in detection_nodes
    ]

    scores_sorted = all(
        detection_scores[index]
        >= detection_scores[index + 1]
        for index in range(
            len(detection_scores) - 1
        )
    )

    node_tensors_finite = all(
        bool(
            torch.isfinite(
                node.raw_node_feature
            ).all()
        )
        and bool(
            torch.isfinite(
                node.attention_profile
            ).all()
        )
        for node in output.nodes
    )

    source_consistent = all(
        (
            node.source == "attention"
            and node.detection_id == -1
            and node.class_id == -1
            and node.class_name
            == "__attention__"
            and len(
                node.attention_region_ids
            ) == 1
        )
        or (
            node.source == "fused"
            and node.detection_id >= 0
            and len(
                node.attention_region_ids
            ) >= 1
        )
        or (
            node.source == "detection"
            and node.detection_id >= 0
            and len(
                node.attention_region_ids
            ) == 0
        )
        for node in output.nodes
    )

    match_ranges_valid = all(
        0.0 <= match.iou <= 1.0
        and 0.0
        <= match.detection_coverage
        <= 1.0
        and 0.0
        <= match.region_coverage
        <= 1.0
        and 0.0
        <= match.overlap_score
        <= 1.0
        for match in output.matches
    )

    raw_dimension = (
        visual_dimension
        + output.geometry_dimension
        + output.scalar_dimension
    )

    loss = torch.zeros(
        (),
        dtype=torch.float32,
    )

    for node in output.nodes:
        loss = (
            loss
            + node.raw_node_feature
            .square()
            .mean()
            + node.attention_profile
            .square()
            .mean()
        )

    loss.backward()

    checks = {
        "input_detection_count": (
            len(detections) == 30
        ),
        "attention_region_count": (
            1 <= len(attention_regions) <= 4
        ),
        "node_count": (
            1 <= len(output.nodes) <= 12
        ),
        "detection_node_limit": (
            len(detection_nodes) <= 10
        ),
        "attention_node_limit": (
            len(attention_nodes) <= 2
        ),
        "sequential_node_ids": (
            tuple(
                node.node_id
                for node in output.nodes
            )
            == tuple(range(len(output.nodes)))
        ),
        "person_retained": (
            selected_classes["person"] >= 1
        ),
        "baseball_bat_retained": (
            selected_classes[
                "baseball bat"
            ] >= 1
        ),
        "correct_bat_detection_retained": (
            4 in selected_detection_ids
        ),
        "fused_node_exists": (
            len(fused_nodes) >= 1
        ),
        "class_limit": all(
            count <= 4
            for count
            in selected_classes.values()
        ),
        "duplicate_free": duplicate_free,
        "scores_sorted": scores_sorted,
        "detection_partition": (
            selected_detection_ids
            .isdisjoint(
                suppressed_detection_ids
            )
            and (
                selected_detection_ids
                | suppressed_detection_ids
            )
            == all_detection_ids
        ),
        "region_partition": (
            retained_region_ids
            .isdisjoint(
                discarded_region_ids
            )
            and retained_region_ids
            .isdisjoint(
                consumed_region_ids
            )
            and discarded_region_ids
            .isdisjoint(
                consumed_region_ids
            )
            and (
                retained_region_ids
                | discarded_region_ids
                | consumed_region_ids
            )
            == all_region_ids
        ),
        "source_consistent": (
            source_consistent
        ),
        "match_ranges": (
            match_ranges_valid
        ),
        "visual_dimension": (
            output.visual_dimension
            == visual_dimension
        ),
        "profile_shape": (
            output.attention_profile_shape
            == profile_shape
        ),
        "geometry_dimension": (
            output.geometry_dimension == 6
        ),
        "scalar_dimension": (
            output.scalar_dimension == 9
        ),
        "raw_feature_shape": all(
            tuple(
                node.raw_node_feature.shape
            )
            == (raw_dimension,)
            for node in output.nodes
        ),
        "profile_output_shape": all(
            tuple(
                node.attention_profile.shape
            )
            == profile_shape
            for node in output.nodes
        ),
        "finite_outputs": (
            node_tensors_finite
        ),
        "visual_gradient": bool(
            projected_visual_features.grad
            is not None
            and torch.isfinite(
                projected_visual_features.grad
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
            and attention_profiles
            .grad
            .abs()
            .sum()
            .item() > 0.0
        ),
        "no_trainable_parameters": (
            sum(
                parameter.numel()
                for parameter
                in builder.parameters()
            )
            == 0
        ),
    }

    print(
        "Hybrid Node Builder summary:"
    )
    print(
        "  Input detections:",
        len(detections),
    )
    print(
        "  Attention Regions:",
        len(attention_regions),
    )
    print(
        "  Retained nodes:",
        len(output.nodes),
    )
    print(
        "  Source counts:",
        dict(source_counts),
    )
    print(
        "  Class counts:",
        dict(selected_classes),
    )
    print(
        "  Matches:",
        len(output.matches),
    )
    print(
        "  Suppressed detections:",
        len(
            output.suppressed_detection_ids
        ),
    )
    print(
        "  Discarded Attention Regions:",
        output.discarded_attention_region_ids,
    )

    print("\nRetained nodes:")

    for node in output.nodes:
        print(
            f"  N{node.node_id:02d} "
            f"source={node.source:<9} "
            f"class={node.class_name:<16} "
            f"det={node.detection_id:>3} "
            f"conf={node.confidence:.3f} "
            f"score={node.selection_score:.3f} "
            f"mass={node.attention_mass:.4f} "
            f"peak={node.attention_peak:.4f} "
            f"regions={node.attention_region_ids}"
        )

    print("\nSelected matches:")

    if output.matches:
        for match in output.matches:
            print(
                f"  detection="
                f"{match.detection_id}, "
                f"region="
                f"A{match.attention_region_id}, "
                f"IoU={match.iou:.3f}, "
                f"det_cov="
                f"{match.detection_coverage:.3f}, "
                f"region_cov="
                f"{match.region_coverage:.3f}, "
                f"score="
                f"{match.overlap_score:.3f}"
            )
    else:
        print("  None")

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
            "Hybrid Node Builder test "
            f"failed: {failed}"
        )

    artifact_path = (
        PROJECT_ROOT
        / "vlm/outputs/"
        "object_graph_visualizations/"
        "baseball_2337002/"
        "hybrid_nodes_test.pt"
    )

    artifact_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "nodes": output.nodes,
            "matches": output.matches,
            "image_size": image_size,
            "tile_grid": tile_grid,
            "selected_detection_ids": (
                output.selected_detection_ids
            ),
            "retained_attention_region_ids": (
                output
                .retained_attention_region_ids
            ),
        },
        artifact_path,
    )

    print(
        "\nSaved Hybrid Nodes:",
        artifact_path,
    )

    print(
        "\nHybrid Node Builder test: PASS"
    )
    print(
        "Note: node selection uses the real "
        "baseball detections and Attention Map, "
        "but synthetic patch features."
    )


if __name__ == "__main__":
    main()