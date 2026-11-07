from hashlib import sha256
from pathlib import Path

from PIL import Image

from vlm.detectors.detection_types import (
    DETECTION_SCHEMA_VERSION,
    DetectionBox,
    ObjectDetectionResult,
)


def file_sha256(
    input_path,
    chunk_size=1024 * 1024,
):
    input_path = Path(input_path)

    digest = sha256()

    with input_path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def normalize_ultralytics_device(device):
    device = str(device).strip().lower()

    if device == "cuda":
        return 0

    if device.startswith("cuda:"):
        return int(
            device.split(":", maxsplit=1)[1]
        )

    return device


def class_name_from_mapping(
    class_mapping,
    class_id,
):
    if isinstance(class_mapping, dict):
        return str(
            class_mapping[int(class_id)]
        )

    return str(
        class_mapping[int(class_id)]
    )


class YOLO11ObjectDetector:
    def __init__(
        self,
        weights_path,
        inference_size=640,
        confidence_threshold=0.25,
        iou_threshold=0.70,
        max_detections=100,
        device="cuda:0",
    ):
        self.weights_path = Path(
            weights_path
        ).expanduser().resolve()

        if not self.weights_path.is_file():
            raise FileNotFoundError(
                "YOLO weights were not found: "
                f"{self.weights_path}"
            )

        self.inference_size = int(
            inference_size
        )

        self.confidence_threshold = float(
            confidence_threshold
        )

        self.iou_threshold = float(
            iou_threshold
        )

        self.max_detections = int(
            max_detections
        )

        self.device = str(device)

        self.ultralytics_device = (
            normalize_ultralytics_device(
                device
            )
        )

        self._validate_configuration()

        try:
            import ultralytics
            from ultralytics import YOLO
        except ImportError as error:
            raise ImportError(
                "Ultralytics is required only "
                "for YOLO detection. Run this "
                "module in the yolo environment."
            ) from error

        self.ultralytics_version = (
            ultralytics.__version__
        )

        self.weights_sha256 = file_sha256(
            self.weights_path
        )

        self.model = YOLO(
            str(self.weights_path)
        )

    def _validate_configuration(self):
        if self.inference_size <= 0:
            raise ValueError(
                "inference_size must be positive."
            )

        if not (
            0.0
            <= self.confidence_threshold
            <= 1.0
        ):
            raise ValueError(
                "confidence_threshold must "
                "be in [0, 1]."
            )

        if not (
            0.0
            <= self.iou_threshold
            <= 1.0
        ):
            raise ValueError(
                "iou_threshold must be in [0, 1]."
            )

        if self.max_detections <= 0:
            raise ValueError(
                "max_detections must be positive."
            )

    def detect(
        self,
        image_path,
        image_id=None,
    ):
        image_path = Path(
            image_path
        ).expanduser().resolve()

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with Image.open(image_path) as image:
            image_width, image_height = (
                image.size
            )

        prediction_results = (
            self.model.predict(
                source=str(image_path),
                imgsz=self.inference_size,
                conf=(
                    self.confidence_threshold
                ),
                iou=self.iou_threshold,
                max_det=self.max_detections,
                device=(
                    self.ultralytics_device
                ),
                save=False,
                verbose=False,
            )
        )

        if len(prediction_results) != 1:
            raise RuntimeError(
                "Expected exactly one YOLO result."
            )

        prediction = prediction_results[0]

        original_height, original_width = (
            prediction.orig_shape
        )

        if (
            int(original_width) != image_width
            or int(original_height)
            != image_height
        ):
            raise RuntimeError(
                "YOLO original image shape "
                "does not match PIL image size."
            )

        boxes = prediction.boxes

        raw_detections = []

        if boxes is not None and len(boxes) > 0:
            xyxy_values = (
                boxes.xyxy
                .detach()
                .to("cpu")
                .float()
                .tolist()
            )

            confidence_values = (
                boxes.conf
                .detach()
                .to("cpu")
                .float()
                .tolist()
            )

            class_values = (
                boxes.cls
                .detach()
                .to("cpu")
                .long()
                .tolist()
            )

            class_mapping = (
                prediction.names
            )

            for (
                coordinates,
                confidence,
                class_id,
            ) in zip(
                xyxy_values,
                confidence_values,
                class_values,
            ):
                x1, y1, x2, y2 = (
                    float(value)
                    for value in coordinates
                )

                x1 = min(
                    max(x1, 0.0),
                    float(image_width),
                )

                y1 = min(
                    max(y1, 0.0),
                    float(image_height),
                )

                x2 = min(
                    max(x2, 0.0),
                    float(image_width),
                )

                y2 = min(
                    max(y2, 0.0),
                    float(image_height),
                )

                if x2 <= x1 or y2 <= y1:
                    continue

                class_name = (
                    class_name_from_mapping(
                        class_mapping,
                        class_id,
                    )
                )

                raw_detections.append(
                    {
                        "class_id": int(
                            class_id
                        ),
                        "class_name": (
                            class_name
                        ),
                        "confidence": float(
                            confidence
                        ),
                        "xyxy": (
                            x1,
                            y1,
                            x2,
                            y2,
                        ),
                    }
                )

        raw_detections.sort(
            key=lambda item: (
                -item["confidence"],
                item["class_id"],
                item["xyxy"],
            )
        )

        detections = []

        for detection_id, item in enumerate(
            raw_detections
        ):
            x1, y1, x2, y2 = item["xyxy"]

            normalized_xyxy = (
                x1 / image_width,
                y1 / image_height,
                x2 / image_width,
                y2 / image_height,
            )

            detections.append(
                DetectionBox(
                    detection_id=detection_id,
                    class_id=(
                        item["class_id"]
                    ),
                    class_name=(
                        item["class_name"]
                    ),
                    confidence=(
                        item["confidence"]
                    ),
                    xyxy=item["xyxy"],
                    normalized_xyxy=(
                        normalized_xyxy
                    ),
                )
            )

        if image_id is None:
            image_id = image_path.stem

        return ObjectDetectionResult(
            schema_version=(
                DETECTION_SCHEMA_VERSION
            ),
            detector_name=(
                "ultralytics_yolo"
            ),
            model_name=(
                self.weights_path.name
            ),
            model_version=(
                self.ultralytics_version
            ),
            weights_sha256=(
                self.weights_sha256
            ),
            image_id=str(image_id),
            image_path=str(image_path),
            image_sha256=file_sha256(
                image_path
            ),
            image_size=(
                image_width,
                image_height,
            ),
            inference_size=(
                self.inference_size
            ),
            confidence_threshold=(
                self.confidence_threshold
            ),
            iou_threshold=(
                self.iou_threshold
            ),
            max_detections=(
                self.max_detections
            ),
            device=self.device,
            detections=tuple(detections),
        )