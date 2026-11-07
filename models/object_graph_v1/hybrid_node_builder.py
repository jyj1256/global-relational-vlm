from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
from torch import nn

from vlm.detectors.detection_types import (
    DetectionBox,
)
from vlm.models.object_graph_v1.attention_region_proposer import (
    AttentionRegion,
)
from vlm.models.object_graph_v1.region_feature_pooler import (
    PooledRegionFeature,
)


@dataclass
class AttentionDetectionMatch:
    detection_id: int
    attention_region_id: int
    iou: float
    detection_coverage: float
    region_coverage: float
    overlap_score: float
    centroid_distance: float


@dataclass
class HybridGraphNode:
    node_id: int
    source: str
    detection_id: int
    attention_region_ids: Tuple[int, ...]
    class_id: int
    class_name: str
    confidence: float
    selection_score: float
    normalized_xyxy: Tuple[
        float,
        float,
        float,
        float,
    ]
    normalized_center: Tuple[float, float]
    normalized_area: float
    attention_mass: float
    attention_density: float
    attention_peak: float
    proposal_score: float
    match_overlap: float
    visual_feature: torch.Tensor
    attention_profile: torch.Tensor
    geometry_feature: torch.Tensor
    scalar_feature: torch.Tensor

    @property
    def raw_node_feature(self):
        return torch.cat(
            [
                self.visual_feature,
                self.geometry_feature,
                self.scalar_feature,
            ],
            dim=-1,
        )


@dataclass
class HybridNodeBuilderOutput:
    nodes: Tuple[HybridGraphNode, ...]
    matches: Tuple[
        AttentionDetectionMatch,
        ...,
    ]
    selected_detection_ids: Tuple[int, ...]
    retained_attention_region_ids: Tuple[
        int,
        ...,
    ]
    suppressed_detection_ids: Tuple[int, ...]
    discarded_attention_region_ids: Tuple[
        int,
        ...,
    ]
    visual_dimension: int
    attention_profile_shape: Tuple[int, ...]
    geometry_dimension: int
    scalar_dimension: int


def _box_area(box):
    x1, y1, x2, y2 = box

    return max(0.0, x2 - x1) * max(
        0.0,
        y2 - y1,
    )


def _box_center(box):
    x1, y1, x2, y2 = box

    return (
        0.5 * (x1 + x2),
        0.5 * (y1 + y2),
    )


def _box_overlap_metrics(
    first_box,
    second_box,
    eps,
):
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

    first_area = _box_area(first_box)
    second_area = _box_area(second_box)
    union = (
        first_area
        + second_area
        - intersection
    )

    iou = intersection / max(union, eps)
    first_coverage = (
        intersection / max(first_area, eps)
    )
    second_coverage = (
        intersection / max(second_area, eps)
    )

    first_center = _box_center(first_box)
    second_center = _box_center(second_box)

    center_distance = (
        (
            first_center[0]
            - second_center[0]
        ) ** 2
        + (
            first_center[1]
            - second_center[1]
        ) ** 2
    ) ** 0.5

    return (
        iou,
        first_coverage,
        second_coverage,
        center_distance,
    )


def _bounded_support(value, scale):
    value = max(float(value), 0.0)

    return value / (value + scale)


class HybridNodeBuilder(nn.Module):
    """
    Build a compact object-attention hybrid node set.

    YOLO detections are the primary nodes. Attention
    Regions either support overlapping detection nodes
    or become residual Attention-only nodes.
    """

    VALID_SOURCES = {
        "detection",
        "fused",
        "attention",
    }

    def __init__(
        self,
        min_detection_confidence=0.20,
        attention_rescue_mass=0.01,
        attention_rescue_peak=0.01,
        min_attention_region_mass=0.01,
        attention_residual_relative_score=0.15,
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
        eps=1e-12,
    ):
        super().__init__()

        probability_values = {
            "min_detection_confidence": (
                min_detection_confidence
            ),
            "attention_rescue_mass": (
                attention_rescue_mass
            ),
            "attention_rescue_peak": (
                attention_rescue_peak
            ),
            "min_attention_region_mass": (
                min_attention_region_mass
            ),
            "attention_residual_relative_score": (
                attention_residual_relative_score
            ),
            "match_overlap_threshold": (
                match_overlap_threshold
            ),
            "duplicate_iou_threshold": (
                duplicate_iou_threshold
            ),
            "max_attention_visual_blend": (
                max_attention_visual_blend
            ),
            "max_region_visual_blend": (
                max_region_visual_blend
            ),
            "attention_only_visual_blend": (
                attention_only_visual_blend
            ),
        }

        for name, value in (
            probability_values.items()
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must lie in [0, 1]."
                )

        integer_values = {
            "max_detection_nodes": (
                max_detection_nodes
            ),
            "max_attention_nodes": (
                max_attention_nodes
            ),
            "max_total_nodes": (
                max_total_nodes
            ),
            "max_nodes_per_class": (
                max_nodes_per_class
            ),
        }

        for name, value in (
            integer_values.items()
        ):
            if int(value) <= 0:
                raise ValueError(
                    f"{name} must be positive."
                )

        if max_total_nodes < 1:
            raise ValueError(
                "max_total_nodes must be positive."
            )

        if attention_support_scale <= 0.0:
            raise ValueError(
                "attention_support_scale must "
                "be positive."
            )

        if proposal_score_scale <= 0.0:
            raise ValueError(
                "proposal_score_scale must "
                "be positive."
            )

        if eps <= 0.0:
            raise ValueError(
                "eps must be positive."
            )

        self.min_detection_confidence = float(
            min_detection_confidence
        )
        self.attention_rescue_mass = float(
            attention_rescue_mass
        )
        self.attention_rescue_peak = float(
            attention_rescue_peak
        )
        self.min_attention_region_mass = float(
            min_attention_region_mass
        )
        self.attention_residual_relative_score = float(
            attention_residual_relative_score
        )
        self.match_overlap_threshold = float(
            match_overlap_threshold
        )
        self.duplicate_iou_threshold = float(
            duplicate_iou_threshold
        )

        self.max_detection_nodes = int(
            max_detection_nodes
        )
        self.max_attention_nodes = int(
            max_attention_nodes
        )
        self.max_total_nodes = int(
            max_total_nodes
        )
        self.max_nodes_per_class = int(
            max_nodes_per_class
        )

        self.attention_support_scale = float(
            attention_support_scale
        )
        self.proposal_score_scale = float(
            proposal_score_scale
        )
        self.max_attention_visual_blend = float(
            max_attention_visual_blend
        )
        self.max_region_visual_blend = float(
            max_region_visual_blend
        )
        self.attention_only_visual_blend = float(
            attention_only_visual_blend
        )
        self.eps = float(eps)

    @staticmethod
    def _boxes_close(
        first_box,
        second_box,
        tolerance=1e-6,
    ):
        return all(
            abs(float(first) - float(second))
            <= tolerance
            for first, second
            in zip(first_box, second_box)
        )

    @staticmethod
    def _feature_shape(feature):
        return (
            int(
                feature
                .geometric_visual_feature
                .shape[-1]
            ),
            tuple(
                int(value)
                for value
                in feature
                .pooled_attention_profile
                .shape
            ),
        )

    def _validate_inputs(
        self,
        detections,
        detection_features,
        attention_regions,
        attention_region_features,
    ):
        if len(detections) != len(
            detection_features
        ):
            raise ValueError(
                "detections and detection_features "
                "must have equal lengths."
            )

        if len(attention_regions) != len(
            attention_region_features
        ):
            raise ValueError(
                "attention_regions and "
                "attention_region_features must "
                "have equal lengths."
            )

        detection_ids = [
            int(detection.detection_id)
            for detection in detections
        ]
        region_ids = [
            int(region.region_id)
            for region in attention_regions
        ]

        if len(set(detection_ids)) != len(
            detection_ids
        ):
            raise ValueError(
                "Detection IDs must be unique."
            )

        if len(set(region_ids)) != len(
            region_ids
        ):
            raise ValueError(
                "Attention Region IDs must "
                "be unique."
            )

        feature_shapes = []

        for detection, feature in zip(
            detections,
            detection_features,
        ):
            if not self._boxes_close(
                detection.normalized_xyxy,
                feature.normalized_xyxy,
            ):
                raise ValueError(
                    "Detection and pooled feature "
                    "boxes do not match."
                )

            feature_shapes.append(
                self._feature_shape(feature)
            )

        for region, feature in zip(
            attention_regions,
            attention_region_features,
        ):
            if not self._boxes_close(
                region.normalized_xyxy,
                feature.normalized_xyxy,
            ):
                raise ValueError(
                    "Attention Region and pooled "
                    "feature boxes do not match."
                )

            feature_shapes.append(
                self._feature_shape(feature)
            )

        if not feature_shapes:
            raise ValueError(
                "At least one Detection or "
                "Attention Region is required."
            )

        if len(set(feature_shapes)) != 1:
            raise ValueError(
                "All pooled features must use "
                "the same dimensions."
            )

        for feature in (
            list(detection_features)
            + list(attention_region_features)
        ):
            tensors = (
                feature.geometric_visual_feature,
                feature.attention_visual_feature,
                feature.pooled_attention_profile,
            )

            if not all(
                bool(torch.isfinite(tensor).all())
                for tensor in tensors
            ):
                raise ValueError(
                    "A pooled feature contains "
                    "NaN or Inf."
                )

        return feature_shapes[0]

    def _build_matches(
        self,
        detections,
        attention_regions,
    ):
        matches = []

        for detection in detections:
            for region in attention_regions:
                (
                    iou,
                    detection_coverage,
                    region_coverage,
                    center_distance,
                ) = _box_overlap_metrics(
                    detection.normalized_xyxy,
                    region.normalized_xyxy,
                    self.eps,
                )

                balanced_coverage = (
                    detection_coverage
                    * region_coverage
                ) ** 0.5

                overlap_score = max(
                    iou,
                    balanced_coverage,
                )

                if (
                    overlap_score
                    < self.match_overlap_threshold
                ):
                    continue

                matches.append(
                    AttentionDetectionMatch(
                        detection_id=int(
                            detection.detection_id
                        ),
                        attention_region_id=int(
                            region.region_id
                        ),
                        iou=float(iou),
                        detection_coverage=float(
                            detection_coverage
                        ),
                        region_coverage=float(
                            region_coverage
                        ),
                        overlap_score=float(
                            overlap_score
                        ),
                        centroid_distance=float(
                            center_distance
                        ),
                    )
                )

        return tuple(matches)

    def _matches_by_detection(
        self,
        matches,
    ):
        grouped = {}

        for match in matches:
            grouped.setdefault(
                match.detection_id,
                [],
            ).append(match)

        for detection_id in grouped:
            grouped[detection_id].sort(
                key=lambda match: (
                    -match.overlap_score,
                    match.attention_region_id,
                )
            )

        return grouped

    def _detection_selection_score(
        self,
        detection,
        feature,
        detection_matches,
        region_lookup,
    ):
        attention_support = _bounded_support(
            feature.attention_mass,
            self.attention_support_scale,
        )

        proposal_support = 0.0

        for match in detection_matches:
            region = region_lookup[
                match.attention_region_id
            ]

            bounded_proposal = _bounded_support(
                region.proposal_score,
                self.proposal_score_scale,
            )

            proposal_support = max(
                proposal_support,
                match.overlap_score
                * bounded_proposal,
            )

        score = (
            0.55 * float(
                detection.confidence
            )
            + 0.35 * attention_support
            + 0.10 * proposal_support
        )

        return float(score)

    def _select_detections(
        self,
        detections,
        detection_features,
        grouped_matches,
        region_lookup,
    ):
        candidates = []

        for detection, feature in zip(
            detections,
            detection_features,
        ):
            detection_matches = (
                grouped_matches.get(
                    int(detection.detection_id),
                    [],
                )
            )

            eligible = (
                detection.confidence
                >= self.min_detection_confidence
                or feature.attention_mass
                >= self.attention_rescue_mass
                or feature.attention_peak
                >= self.attention_rescue_peak
            )

            if not eligible:
                continue

            score = (
                self._detection_selection_score(
                    detection=detection,
                    feature=feature,
                    detection_matches=(
                        detection_matches
                    ),
                    region_lookup=region_lookup,
                )
            )

            candidates.append(
                (
                    score,
                    detection,
                    feature,
                    tuple(detection_matches),
                )
            )

        candidates.sort(
            key=lambda item: (
                -item[0],
                -float(item[1].confidence),
                int(item[1].detection_id),
            )
        )

        selected = []
        class_counts = {}

        for candidate in candidates:
            if len(selected) >= (
                self.max_detection_nodes
            ):
                break

            (
                score,
                detection,
                feature,
                detection_matches,
            ) = candidate

            class_id = int(
                detection.class_id
            )

            if class_counts.get(class_id, 0) >= (
                self.max_nodes_per_class
            ):
                continue

            duplicate = False

            for selected_item in selected:
                selected_detection = (
                    selected_item[1]
                )

                if (
                    int(selected_detection.class_id)
                    != class_id
                ):
                    continue

                iou, _, _, _ = (
                    _box_overlap_metrics(
                        detection.normalized_xyxy,
                        selected_detection
                        .normalized_xyxy,
                        self.eps,
                    )
                )

                if (
                    iou
                    >= self.duplicate_iou_threshold
                ):
                    duplicate = True
                    break

            if duplicate:
                continue

            selected.append(
                (
                    score,
                    detection,
                    feature,
                    detection_matches,
                )
            )
            class_counts[class_id] = (
                class_counts.get(
                    class_id,
                    0,
                )
                + 1
            )

        return selected

    def _geometry_feature(
        self,
        box,
        device,
    ):
        x1, y1, x2, y2 = box
        width = max(x2 - x1, self.eps)
        height = max(y2 - y1, self.eps)
        center_x, center_y = (
            _box_center(box)
        )
        area = width * height

        log_aspect = torch.log(
            torch.tensor(
                width / height,
                device=device,
                dtype=torch.float32,
            )
        )

        normalized_log_aspect = (
            log_aspect.clamp(
                min=-4.0,
                max=4.0,
            )
            / 4.0
        )

        return torch.stack(
            [
                torch.tensor(
                    center_x,
                    device=device,
                    dtype=torch.float32,
                ),
                torch.tensor(
                    center_y,
                    device=device,
                    dtype=torch.float32,
                ),
                torch.tensor(
                    width,
                    device=device,
                    dtype=torch.float32,
                ),
                torch.tensor(
                    height,
                    device=device,
                    dtype=torch.float32,
                ),
                torch.tensor(
                    area,
                    device=device,
                    dtype=torch.float32,
                ),
                normalized_log_aspect,
            ]
        )

    def _scalar_feature(
        self,
        source,
        confidence,
        attention_mass,
        attention_density,
        attention_peak,
        proposal_score,
        match_overlap,
        device,
    ):
        source_flags = {
            "detection": (
                1.0,
                0.0,
                0.0,
            ),
            "fused": (
                0.0,
                1.0,
                0.0,
            ),
            "attention": (
                0.0,
                0.0,
                1.0,
            ),
        }

        flags = source_flags[source]

        values = [
            float(confidence),
            float(attention_mass),
            _bounded_support(
                attention_density,
                1.0,
            ),
            float(attention_peak),
            _bounded_support(
                proposal_score,
                self.proposal_score_scale,
            ),
            float(match_overlap),
            *flags,
        ]

        return torch.tensor(
            values,
            device=device,
            dtype=torch.float32,
        )

    def _build_detection_node(
        self,
        node_id,
        score,
        detection,
        feature,
        detection_matches,
        region_lookup,
        region_feature_lookup,
    ):
        geometric_visual = (
            feature
            .geometric_visual_feature
            .float()
        )
        attention_visual = (
            feature
            .attention_visual_feature
            .to(geometric_visual.device)
            .float()
        )

        attention_support = _bounded_support(
            feature.attention_mass,
            self.attention_support_scale,
        )

        attention_blend = (
            self.max_attention_visual_blend
            * attention_support
        )

        visual_feature = (
            geometric_visual
            + attention_blend
            * (
                attention_visual
                - geometric_visual
            )
        )

        attention_profile = (
            feature
            .pooled_attention_profile
            .to(geometric_visual.device)
            .float()
        )

        proposal_score = 0.0
        match_overlap = 0.0
        matched_region_ids = []

        if detection_matches:
            region_visuals = []
            region_profiles = []
            region_weights = []

            for match in detection_matches:
                region = region_lookup[
                    match.attention_region_id
                ]
                region_feature = (
                    region_feature_lookup[
                        match.attention_region_id
                    ]
                )

                matched_region_ids.append(
                    int(region.region_id)
                )

                proposal_score = max(
                    proposal_score,
                    float(
                        region.proposal_score
                    ),
                )
                match_overlap = max(
                    match_overlap,
                    float(
                        match.overlap_score
                    ),
                )

                weight = (
                    max(
                        float(
                            match.overlap_score
                        ),
                        self.eps,
                    )
                    * max(
                        float(
                            region.proposal_score
                        ),
                        self.eps,
                    )
                )

                region_weights.append(weight)
                region_visuals.append(
                    region_feature
                    .attention_visual_feature
                    .to(
                        geometric_visual.device
                    )
                    .float()
                )
                region_profiles.append(
                    region_feature
                    .pooled_attention_profile
                    .to(
                        geometric_visual.device
                    )
                    .float()
                )

            weights = torch.tensor(
                region_weights,
                device=geometric_visual.device,
                dtype=torch.float32,
            )
            weights = (
                weights
                / weights.sum().clamp_min(
                    self.eps
                )
            )

            stacked_visuals = torch.stack(
                region_visuals,
                dim=0,
            )
            stacked_profiles = torch.stack(
                region_profiles,
                dim=0,
            )

            region_visual = torch.sum(
                stacked_visuals
                * weights.unsqueeze(-1),
                dim=0,
            )

            profile_weight_shape = (
                weights.shape[0],
                *(
                    1
                    for _ in
                    stacked_profiles.shape[1:]
                ),
            )

            region_profile = torch.sum(
                stacked_profiles
                * weights.reshape(
                    profile_weight_shape
                ),
                dim=0,
            )

            region_blend = (
                self.max_region_visual_blend
                * match_overlap
            )

            visual_feature = (
                visual_feature
                + region_blend
                * (
                    region_visual
                    - visual_feature
                )
            )

            attention_profile = (
                attention_profile
                + region_blend
                * (
                    region_profile
                    - attention_profile
                )
            )

        source = (
            "fused"
            if matched_region_ids
            else "detection"
        )

        box = tuple(
            float(value)
            for value
            in detection.normalized_xyxy
        )
        center = _box_center(box)
        area = _box_area(box)

        geometry_feature = (
            self._geometry_feature(
                box=box,
                device=visual_feature.device,
            )
        )
        scalar_feature = (
            self._scalar_feature(
                source=source,
                confidence=(
                    detection.confidence
                ),
                attention_mass=(
                    feature.attention_mass
                ),
                attention_density=(
                    feature.attention_density
                ),
                attention_peak=(
                    feature.attention_peak
                ),
                proposal_score=proposal_score,
                match_overlap=match_overlap,
                device=visual_feature.device,
            )
        )

        return HybridGraphNode(
            node_id=int(node_id),
            source=source,
            detection_id=int(
                detection.detection_id
            ),
            attention_region_ids=tuple(
                sorted(
                    set(matched_region_ids)
                )
            ),
            class_id=int(
                detection.class_id
            ),
            class_name=str(
                detection.class_name
            ),
            confidence=float(
                detection.confidence
            ),
            selection_score=float(score),
            normalized_xyxy=box,
            normalized_center=(
                float(center[0]),
                float(center[1]),
            ),
            normalized_area=float(area),
            attention_mass=float(
                feature.attention_mass
            ),
            attention_density=float(
                feature.attention_density
            ),
            attention_peak=float(
                feature.attention_peak
            ),
            proposal_score=float(
                proposal_score
            ),
            match_overlap=float(
                match_overlap
            ),
            visual_feature=visual_feature,
            attention_profile=(
                attention_profile
            ),
            geometry_feature=(
                geometry_feature
            ),
            scalar_feature=scalar_feature,
        )

    def _build_attention_node(
        self,
        node_id,
        score,
        region,
        feature,
    ):
        geometric_visual = (
            feature
            .geometric_visual_feature
            .float()
        )
        attention_visual = (
            feature
            .attention_visual_feature
            .to(geometric_visual.device)
            .float()
        )

        visual_feature = (
            geometric_visual
            + self.attention_only_visual_blend
            * (
                attention_visual
                - geometric_visual
            )
        )

        attention_profile = (
            feature
            .pooled_attention_profile
            .to(geometric_visual.device)
            .float()
        )

        box = tuple(
            float(value)
            for value
            in region.normalized_xyxy
        )
        center = _box_center(box)
        area = _box_area(box)

        geometry_feature = (
            self._geometry_feature(
                box=box,
                device=visual_feature.device,
            )
        )
        scalar_feature = (
            self._scalar_feature(
                source="attention",
                confidence=0.0,
                attention_mass=(
                    feature.attention_mass
                ),
                attention_density=(
                    feature.attention_density
                ),
                attention_peak=(
                    feature.attention_peak
                ),
                proposal_score=(
                    region.proposal_score
                ),
                match_overlap=0.0,
                device=visual_feature.device,
            )
        )

        return HybridGraphNode(
            node_id=int(node_id),
            source="attention",
            detection_id=-1,
            attention_region_ids=(
                int(region.region_id),
            ),
            class_id=-1,
            class_name="__attention__",
            confidence=0.0,
            selection_score=float(score),
            normalized_xyxy=box,
            normalized_center=(
                float(center[0]),
                float(center[1]),
            ),
            normalized_area=float(area),
            attention_mass=float(
                feature.attention_mass
            ),
            attention_density=float(
                feature.attention_density
            ),
            attention_peak=float(
                feature.attention_peak
            ),
            proposal_score=float(
                region.proposal_score
            ),
            match_overlap=0.0,
            visual_feature=visual_feature,
            attention_profile=(
                attention_profile
            ),
            geometry_feature=(
                geometry_feature
            ),
            scalar_feature=scalar_feature,
        )

    def forward(
        self,
        detections: Sequence[DetectionBox],
        detection_features: Sequence[
            PooledRegionFeature
        ],
        attention_regions: Sequence[
            AttentionRegion
        ],
        attention_region_features: Sequence[
            PooledRegionFeature
        ],
    ) -> HybridNodeBuilderOutput:
        (
            visual_dimension,
            profile_shape,
        ) = self._validate_inputs(
            detections=detections,
            detection_features=(
                detection_features
            ),
            attention_regions=(
                attention_regions
            ),
            attention_region_features=(
                attention_region_features
            ),
        )

        region_lookup = {
            int(region.region_id): region
            for region in attention_regions
        }
        region_feature_lookup = {
            int(region.region_id): feature
            for region, feature in zip(
                attention_regions,
                attention_region_features,
            )
        }

        all_matches = self._build_matches(
            detections=detections,
            attention_regions=attention_regions,
        )

        grouped_matches = (
            self._matches_by_detection(
                all_matches
            )
        )

        selected_detections = (
            self._select_detections(
                detections=detections,
                detection_features=(
                    detection_features
                ),
                grouped_matches=(
                    grouped_matches
                ),
                region_lookup=region_lookup,
            )
        )

        nodes = []
        selected_detection_ids = []
        selected_matches = []

        for (
            score,
            detection,
            feature,
            detection_matches,
        ) in selected_detections:
            if len(nodes) >= (
                self.max_total_nodes
            ):
                break

            node = (
                self._build_detection_node(
                    node_id=len(nodes),
                    score=score,
                    detection=detection,
                    feature=feature,
                    detection_matches=(
                        detection_matches
                    ),
                    region_lookup=region_lookup,
                    region_feature_lookup=(
                        region_feature_lookup
                    ),
                )
            )

            nodes.append(node)
            selected_detection_ids.append(
                int(detection.detection_id)
            )
            selected_matches.extend(
                detection_matches
            )

        consumed_region_ids = {
            int(match.attention_region_id)
            for match in selected_matches
        }

        attention_candidates = []

        max_region_proposal_score = max(
            (
                float(region.proposal_score)
                for region in attention_regions
            ),
            default=0.0,
        )

        minimum_residual_proposal_score = (
            max_region_proposal_score
            * self.attention_residual_relative_score
        )

        for region, feature in zip(
            attention_regions,
            attention_region_features,
        ):
            region_id = int(region.region_id)

            if region_id in consumed_region_ids:
                continue

            if (
                float(region.proposal_score)
                < minimum_residual_proposal_score
            ):
                continue

            if (
                feature.attention_mass
                < self.min_attention_region_mass
            ):
                continue

            proposal_support = _bounded_support(
                region.proposal_score,
                self.proposal_score_scale,
            )
            attention_support = _bounded_support(
                feature.attention_mass,
                self.attention_support_scale,
            )

            score = (
                0.70 * proposal_support
                + 0.30 * attention_support
            )

            attention_candidates.append(
                (
                    float(score),
                    region,
                    feature,
                )
            )

        attention_candidates.sort(
            key=lambda item: (
                -item[0],
                -float(
                    item[1].attention_mass
                ),
                int(item[1].region_id),
            )
        )

        retained_attention_region_ids = []

        for (
            score,
            region,
            feature,
        ) in attention_candidates:
            if (
                len(
                    retained_attention_region_ids
                )
                >= self.max_attention_nodes
            ):
                break

            if len(nodes) >= (
                self.max_total_nodes
            ):
                break

            node = (
                self._build_attention_node(
                    node_id=len(nodes),
                    score=score,
                    region=region,
                    feature=feature,
                )
            )

            nodes.append(node)
            retained_attention_region_ids.append(
                int(region.region_id)
            )

        if not nodes:
            raise RuntimeError(
                "No Hybrid Graph nodes were retained."
            )

        selected_detection_id_set = set(
            selected_detection_ids
        )
        retained_region_id_set = set(
            retained_attention_region_ids
        )
        consumed_region_id_set = set(
            consumed_region_ids
        )

        suppressed_detection_ids = tuple(
            int(detection.detection_id)
            for detection in detections
            if int(detection.detection_id)
            not in selected_detection_id_set
        )

        discarded_attention_region_ids = tuple(
            int(region.region_id)
            for region in attention_regions
            if (
                int(region.region_id)
                not in retained_region_id_set
                and int(region.region_id)
                not in consumed_region_id_set
            )
        )

        selected_matches = tuple(
            sorted(
                selected_matches,
                key=lambda match: (
                    match.detection_id,
                    match.attention_region_id,
                ),
            )
        )

        for node in nodes:
            tensors = (
                node.visual_feature,
                node.attention_profile,
                node.geometry_feature,
                node.scalar_feature,
                node.raw_node_feature,
            )

            if not all(
                bool(torch.isfinite(tensor).all())
                for tensor in tensors
            ):
                raise RuntimeError(
                    "A Hybrid Graph node contains "
                    "NaN or Inf."
                )

        return HybridNodeBuilderOutput(
            nodes=tuple(nodes),
            matches=selected_matches,
            selected_detection_ids=tuple(
                selected_detection_ids
            ),
            retained_attention_region_ids=(
                tuple(
                    retained_attention_region_ids
                )
            ),
            suppressed_detection_ids=(
                suppressed_detection_ids
            ),
            discarded_attention_region_ids=(
                discarded_attention_region_ids
            ),
            visual_dimension=int(
                visual_dimension
            ),
            attention_profile_shape=tuple(
                profile_shape
            ),
            geometry_dimension=6,
            scalar_dimension=9,
        )