from dataclasses import dataclass
from math import isfinite
from typing import Any, Dict, Tuple


DETECTION_SCHEMA_VERSION = 1

BoxCoordinates = Tuple[
    float,
    float,
    float,
    float,
]


def _as_box_tuple(
    values,
    name,
) -> BoxCoordinates:
    if len(values) != 4:
        raise ValueError(
            f"{name} must contain four values."
        )

    coordinates = tuple(
        float(value)
        for value in values
    )

    if not all(
        isfinite(value)
        for value in coordinates
    ):
        raise ValueError(
            f"{name} contains a non-finite value."
        )

    return coordinates


def _validate_unit_interval(
    value,
    name,
):
    if not isfinite(value):
        raise ValueError(
            f"{name} must be finite."
        )

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be in [0, 1]."
        )


def _validate_sha256(
    value,
    name,
):
    if len(value) != 64:
        raise ValueError(
            f"{name} must contain 64 characters."
        )

    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(
            f"{name} must be hexadecimal."
        ) from error


@dataclass(frozen=True)
class DetectionBox:
    detection_id: int
    class_id: int
    class_name: str
    confidence: float
    xyxy: BoxCoordinates
    normalized_xyxy: BoxCoordinates

    def __post_init__(self):
        object.__setattr__(
            self,
            "detection_id",
            int(self.detection_id),
        )

        object.__setattr__(
            self,
            "class_id",
            int(self.class_id),
        )

        object.__setattr__(
            self,
            "class_name",
            str(self.class_name).strip(),
        )

        object.__setattr__(
            self,
            "confidence",
            float(self.confidence),
        )

        object.__setattr__(
            self,
            "xyxy",
            _as_box_tuple(
                self.xyxy,
                "xyxy",
            ),
        )

        object.__setattr__(
            self,
            "normalized_xyxy",
            _as_box_tuple(
                self.normalized_xyxy,
                "normalized_xyxy",
            ),
        )

        self.validate()

    def validate(self):
        if self.detection_id < 0:
            raise ValueError(
                "detection_id cannot be negative."
            )

        if self.class_id < 0:
            raise ValueError(
                "class_id cannot be negative."
            )

        if not self.class_name:
            raise ValueError(
                "class_name must not be empty."
            )

        _validate_unit_interval(
            self.confidence,
            "confidence",
        )

        x1, y1, x2, y2 = self.xyxy

        if x1 < 0.0 or y1 < 0.0:
            raise ValueError(
                "Absolute box coordinates "
                "cannot be negative."
            )

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                "xyxy must define a positive-area box."
            )

        (
            normalized_x1,
            normalized_y1,
            normalized_x2,
            normalized_y2,
        ) = self.normalized_xyxy

        for index, value in enumerate(
            self.normalized_xyxy
        ):
            _validate_unit_interval(
                value,
                f"normalized_xyxy[{index}]",
            )

        if (
            normalized_x2 <= normalized_x1
            or normalized_y2 <= normalized_y1
        ):
            raise ValueError(
                "normalized_xyxy must define "
                "a positive-area box."
            )

    @property
    def width(self):
        return self.xyxy[2] - self.xyxy[0]

    @property
    def height(self):
        return self.xyxy[3] - self.xyxy[1]

    @property
    def area(self):
        return self.width * self.height

    @property
    def normalized_area(self):
        box = self.normalized_xyxy

        return (
            (box[2] - box[0])
            * (box[3] - box[1])
        )

    @property
    def normalized_center(self):
        box = self.normalized_xyxy

        return (
            (box[0] + box[2]) / 2.0,
            (box[1] + box[3]) / 2.0,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "xyxy": list(self.xyxy),
            "normalized_xyxy": list(
                self.normalized_xyxy
            ),
        }

    @classmethod
    def from_dict(
        cls,
        payload,
    ):
        return cls(
            detection_id=payload["detection_id"],
            class_id=payload["class_id"],
            class_name=payload["class_name"],
            confidence=payload["confidence"],
            xyxy=payload["xyxy"],
            normalized_xyxy=(
                payload["normalized_xyxy"]
            ),
        )


@dataclass(frozen=True)
class ObjectDetectionResult:
    schema_version: int
    detector_name: str
    model_name: str
    model_version: str
    weights_sha256: str
    image_id: str
    image_path: str
    image_sha256: str
    image_size: Tuple[int, int]
    inference_size: int
    confidence_threshold: float
    iou_threshold: float
    max_detections: int
    device: str
    detections: Tuple[
        DetectionBox,
        ...,
    ]

    def __post_init__(self):
        object.__setattr__(
            self,
            "schema_version",
            int(self.schema_version),
        )

        object.__setattr__(
            self,
            "detector_name",
            str(self.detector_name).strip(),
        )

        object.__setattr__(
            self,
            "model_name",
            str(self.model_name).strip(),
        )

        object.__setattr__(
            self,
            "model_version",
            str(self.model_version).strip(),
        )

        object.__setattr__(
            self,
            "weights_sha256",
            str(self.weights_sha256).lower(),
        )

        object.__setattr__(
            self,
            "image_id",
            str(self.image_id).strip(),
        )

        object.__setattr__(
            self,
            "image_path",
            str(self.image_path),
        )

        object.__setattr__(
            self,
            "image_sha256",
            str(self.image_sha256).lower(),
        )

        object.__setattr__(
            self,
            "image_size",
            tuple(
                int(value)
                for value in self.image_size
            ),
        )

        object.__setattr__(
            self,
            "inference_size",
            int(self.inference_size),
        )

        object.__setattr__(
            self,
            "confidence_threshold",
            float(self.confidence_threshold),
        )

        object.__setattr__(
            self,
            "iou_threshold",
            float(self.iou_threshold),
        )

        object.__setattr__(
            self,
            "max_detections",
            int(self.max_detections),
        )

        object.__setattr__(
            self,
            "device",
            str(self.device),
        )

        object.__setattr__(
            self,
            "detections",
            tuple(
                (
                    detection
                    if isinstance(
                        detection,
                        DetectionBox,
                    )
                    else DetectionBox.from_dict(
                        detection
                    )
                )
                for detection
                in self.detections
            ),
        )

        self.validate()

    def validate(self):
        if (
            self.schema_version
            != DETECTION_SCHEMA_VERSION
        ):
            raise ValueError(
                "Unsupported detection schema."
            )

        for name, value in (
            (
                "detector_name",
                self.detector_name,
            ),
            (
                "model_name",
                self.model_name,
            ),
            (
                "model_version",
                self.model_version,
            ),
            (
                "image_id",
                self.image_id,
            ),
            (
                "image_path",
                self.image_path,
            ),
            (
                "device",
                self.device,
            ),
        ):
            if not value:
                raise ValueError(
                    f"{name} must not be empty."
                )

        _validate_sha256(
            self.weights_sha256,
            "weights_sha256",
        )

        _validate_sha256(
            self.image_sha256,
            "image_sha256",
        )

        if len(self.image_size) != 2:
            raise ValueError(
                "image_size must be (width, height)."
            )

        image_width, image_height = (
            self.image_size
        )

        if image_width <= 0 or image_height <= 0:
            raise ValueError(
                "image_size must be positive."
            )

        if self.inference_size <= 0:
            raise ValueError(
                "inference_size must be positive."
            )

        _validate_unit_interval(
            self.confidence_threshold,
            "confidence_threshold",
        )

        _validate_unit_interval(
            self.iou_threshold,
            "iou_threshold",
        )

        if self.max_detections <= 0:
            raise ValueError(
                "max_detections must be positive."
            )

        detection_ids = [
            detection.detection_id
            for detection in self.detections
        ]

        if len(detection_ids) != len(
            set(detection_ids)
        ):
            raise ValueError(
                "detection_id values must be unique."
            )

        for detection in self.detections:
            x1, y1, x2, y2 = detection.xyxy

            if (
                x2 > image_width + 1e-3
                or y2 > image_height + 1e-3
            ):
                raise ValueError(
                    "Detection box exceeds "
                    "the image boundary."
                )

    def class_counts(self):
        counts = {}

        for detection in self.detections:
            counts[detection.class_name] = (
                counts.get(
                    detection.class_name,
                    0,
                )
                + 1
            )

        return counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "detector_name": (
                self.detector_name
            ),
            "model_name": self.model_name,
            "model_version": (
                self.model_version
            ),
            "weights_sha256": (
                self.weights_sha256
            ),
            "image_id": self.image_id,
            "image_path": self.image_path,
            "image_sha256": (
                self.image_sha256
            ),
            "image_size": list(
                self.image_size
            ),
            "inference_size": (
                self.inference_size
            ),
            "confidence_threshold": (
                self.confidence_threshold
            ),
            "iou_threshold": (
                self.iou_threshold
            ),
            "max_detections": (
                self.max_detections
            ),
            "device": self.device,
            "detections": [
                detection.to_dict()
                for detection
                in self.detections
            ],
        }

    @classmethod
    def from_dict(
        cls,
        payload,
    ):
        return cls(
            schema_version=(
                payload["schema_version"]
            ),
            detector_name=(
                payload["detector_name"]
            ),
            model_name=payload["model_name"],
            model_version=(
                payload["model_version"]
            ),
            weights_sha256=(
                payload["weights_sha256"]
            ),
            image_id=payload["image_id"],
            image_path=payload["image_path"],
            image_sha256=(
                payload["image_sha256"]
            ),
            image_size=payload["image_size"],
            inference_size=(
                payload["inference_size"]
            ),
            confidence_threshold=(
                payload[
                    "confidence_threshold"
                ]
            ),
            iou_threshold=(
                payload["iou_threshold"]
            ),
            max_detections=(
                payload["max_detections"]
            ),
            device=payload["device"],
            detections=tuple(
                DetectionBox.from_dict(item)
                for item
                in payload["detections"]
            ),
        )