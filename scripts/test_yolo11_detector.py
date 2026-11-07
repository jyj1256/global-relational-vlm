import argparse
import json
import sys
from pathlib import Path

from PIL import (
    Image,
    ImageDraw,
    ImageFont,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from vlm.detectors.detection_types import (
    ObjectDetectionResult,
)
from vlm.detectors.yolo11_detector import (
    YOLO11ObjectDetector,
)


COLOR_PALETTE = (
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a YOLO11s detection smoke "
            "test on the GQA baseball image."
        )
    )

    parser.add_argument(
        "--image",
        type=Path,
        default=(
            PROJECT_ROOT
            / "datasets/gqa/images/"
            / "2337002.jpg"
        ),
    )

    parser.add_argument(
        "--image-id",
        type=str,
        default="2337002",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=(
            PROJECT_ROOT
            / "pretrained/yolo/"
            / "yolo11s.pt"
        ),
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=(
            PROJECT_ROOT
            / "vlm/outputs/"
            / "object_detections/"
            / "yolo11s_coco/entries/"
            / "2337002.json"
        ),
    )

    parser.add_argument(
        "--output-image",
        type=Path,
        default=(
            PROJECT_ROOT
            / "vlm/outputs/"
            / "object_detections/"
            / "yolo11s_coco/"
            / "visualizations/"
            / "2337002.png"
        ),
    )

    parser.add_argument(
        "--inference-size",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.10,
        help=(
            "Low smoke-test threshold used to "
            "inspect small bat/glove candidates. "
            "This is not the final threshold."
        ),
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--max-detections",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    return parser.parse_args()


def load_font(
    size,
    bold=False,
):
    font_names = (
        (
            "DejaVuSans-Bold.ttf",
            "LiberationSans-Bold.ttf",
        )
        if bold
        else (
            "DejaVuSans.ttf",
            "LiberationSans-Regular.ttf",
        )
    )

    for font_name in font_names:
        try:
            return ImageFont.truetype(
                font_name,
                size=size,
            )
        except OSError:
            continue

    return ImageFont.load_default()


def detection_color(
    class_id,
):
    return COLOR_PALETTE[
        int(class_id)
        % len(COLOR_PALETTE)
    ]


def save_detection_visualization(
    image_path,
    detection_result,
    output_path,
):
    image = Image.open(
        image_path
    ).convert("RGB")

    draw = ImageDraw.Draw(image)

    minimum_dimension = min(
        image.size
    )

    line_width = max(
        2,
        round(
            minimum_dimension / 250
        ),
    )

    font_size = max(
        14,
        round(
            minimum_dimension / 32
        ),
    )

    font = load_font(
        size=font_size,
        bold=True,
    )

    for detection in (
        detection_result.detections
    ):
        x1, y1, x2, y2 = (
            detection.xyxy
        )

        color = detection_color(
            detection.class_id
        )

        draw.rectangle(
            (x1, y1, x2, y2),
            outline=color,
            width=line_width,
        )

        label = (
            f"#{detection.detection_id} "
            f"{detection.class_name} "
            f"{detection.confidence:.2f}"
        )

        text_box = draw.textbbox(
            (x1, y1),
            label,
            font=font,
        )

        text_width = (
            text_box[2]
            - text_box[0]
        )

        text_height = (
            text_box[3]
            - text_box[1]
        )

        label_y = max(
            0.0,
            y1 - text_height - 6,
        )

        draw.rectangle(
            (
                x1,
                label_y,
                min(
                    x1 + text_width + 8,
                    image.width,
                ),
                label_y
                + text_height
                + 6,
            ),
            fill=color,
        )

        draw.text(
            (
                x1 + 4,
                label_y + 2,
            ),
            label,
            fill=(0, 0, 0),
            font=font,
        )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(
        output_path,
        format="PNG",
    )


def save_detection_json(
    detection_result,
    output_path,
):
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = Path(
        str(output_path) + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            detection_result.to_dict(),
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

        file.write("\n")

    temporary_path.replace(
        output_path
    )


def load_detection_json(
    input_path,
):
    with Path(input_path).open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    return ObjectDetectionResult.from_dict(
        payload
    )


def main():
    args = parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(
            f"Image not found: {args.image}"
        )

    if not args.model.is_file():
        raise FileNotFoundError(
            f"Model not found: {args.model}"
        )

    print("YOLO11s baseball smoke test:")
    print("  Image:", args.image)
    print("  Model:", args.model)
    print(
        "  Inference size:",
        args.inference_size,
    )
    print(
        "  Confidence threshold:",
        args.confidence,
    )
    print(
        "  IoU threshold:",
        args.iou,
    )
    print(
        "  Maximum detections:",
        args.max_detections,
    )
    print("  Device:", args.device)

    print("\nLoading YOLO11s...")

    detector = YOLO11ObjectDetector(
        weights_path=args.model,
        inference_size=(
            args.inference_size
        ),
        confidence_threshold=(
            args.confidence
        ),
        iou_threshold=args.iou,
        max_detections=(
            args.max_detections
        ),
        device=args.device,
    )

    print("Running object detection...")

    detection_result = detector.detect(
        image_path=args.image,
        image_id=args.image_id,
    )

    save_detection_json(
        detection_result=detection_result,
        output_path=args.output_json,
    )

    save_detection_visualization(
        image_path=args.image,
        detection_result=(
            detection_result
        ),
        output_path=args.output_image,
    )

    restored_result = (
        load_detection_json(
            args.output_json
        )
    )

    class_counts = (
        detection_result.class_counts()
    )

    print("\nDetection summary:")
    print(
        "  Image size:",
        detection_result.image_size,
    )
    print(
        "  Total detections:",
        len(
            detection_result.detections
        ),
    )
    print(
        "  Ultralytics version:",
        detection_result.model_version,
    )
    print(
        "  Model SHA-256:",
        detection_result.weights_sha256,
    )
    print(
        "  Image SHA-256:",
        detection_result.image_sha256,
    )

    print("\nClass counts:")

    for class_name, count in sorted(
        class_counts.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    ):
        print(
            f"  {class_name:<20} "
            f"{count}"
        )

    print("\nFirst 40 detections:")

    for detection in (
        detection_result.detections[:40]
    ):
        print(
            f"  #{detection.detection_id:02d} "
            f"{detection.class_name:<18} "
            f"conf={detection.confidence:.4f} "
            f"area={detection.normalized_area:.6f} "
            f"box=("
            f"{detection.xyxy[0]:.1f}, "
            f"{detection.xyxy[1]:.1f}, "
            f"{detection.xyxy[2]:.1f}, "
            f"{detection.xyxy[3]:.1f})"
        )

    target_classes = (
        "person",
        "baseball bat",
        "baseball glove",
    )

    print("\nBaseball target classes:")

    for class_name in target_classes:
        matching_detections = [
            detection
            for detection
            in detection_result.detections
            if detection.class_name
            == class_name
        ]

        print(
            f"  {class_name:<18} "
            f"count={len(matching_detections)}"
        )

        for detection in (
            matching_detections
        ):
            print(
                f"    "
                f"#{detection.detection_id:02d} "
                f"conf={detection.confidence:.4f} "
                f"center="
                f"("
                f"{detection.normalized_center[0]:.4f}, "
                f"{detection.normalized_center[1]:.4f}"
                f") "
                f"area="
                f"{detection.normalized_area:.6f}"
            )

    all_boxes_valid = all(
        (
            detection.xyxy[2]
            > detection.xyxy[0]
            and detection.xyxy[3]
            > detection.xyxy[1]
            and detection.normalized_area
            > 0.0
        )
        for detection
        in detection_result.detections
    )

    checks = {
        "nonempty_detection": (
            len(
                detection_result.detections
            )
            > 0
        ),
        "person_detected": (
            class_counts.get(
                "person",
                0,
            )
            > 0
        ),
        "boxes_valid": (
            all_boxes_valid
        ),
        "json_round_trip": (
            restored_result.to_dict()
            == detection_result.to_dict()
        ),
        "json_saved": (
            args.output_json.is_file()
        ),
        "visualization_saved": (
            args.output_image.is_file()
        ),
        "model_hash": (
            detection_result.weights_sha256
            == (
                "85a76fe86dd8afe384648546b56a7a"
                "78580c7cb7b404fc595f97969322d"
                "502d5"
            )
        ),
    }

    print("\nChecks:")

    for name, passed in checks.items():
        print(
            f"  {name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    if not all(checks.values()):
        raise RuntimeError(
            "YOLO11s baseball smoke test failed."
        )

    print("\nSaved outputs:")
    print("  JSON:", args.output_json)
    print(
        "  Visualization:",
        args.output_image,
    )

    print(
        "\nYOLO11s baseball "
        "smoke test: PASS"
    )

    print(
        "Note: baseball bat and glove "
        "detections are diagnostic only and "
        "are not required to pass this test."
    )


if __name__ == "__main__":
    main()