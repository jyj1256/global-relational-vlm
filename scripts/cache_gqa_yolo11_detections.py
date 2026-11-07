import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from vlm.data.gqa_dataset import (
    GQADataset,
)
from vlm.detectors.yolo11_detector import (
    YOLO11ObjectDetector,
    file_sha256,
)
from vlm.utils.object_detection_cache import (
    detection_cache_path,
    load_object_detection_cache,
    save_object_detection_cache,
)


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "datasets/gqa/subsets/"
    "train_debug_100_seed2026.jsonl"
)

DEFAULT_DATA_ROOT = (
    PROJECT_ROOT
    / "datasets/gqa"
)

DEFAULT_WEIGHTS = (
    PROJECT_ROOT
    / "pretrained/yolo/yolo11s.pt"
)

DEFAULT_CACHE_ROOT = (
    PROJECT_ROOT
    / "vlm/outputs/object_detections/"
    "yolo11s_coco"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Cache YOLO11s detections for "
            "a GQA JSONL subset."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )

    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
    )

    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
    )

    parser.add_argument(
        "--inference-size",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--iou-threshold",
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

    parser.add_argument(
        "--max-new-images",
        type=int,
        default=0,
        help=(
            "Process at most this many missing "
            "images. Zero means no limit."
        ),
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Regenerate all selected image "
            "caches even when valid caches exist."
        ),
    )

    return parser.parse_args()


def utc_timestamp():
    return datetime.now(
        timezone.utc
    ).isoformat()


def write_json_atomic(
    output_path,
    payload,
):
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = Path(
        str(output_path) + ".tmp"
    )

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )

            file.write("\n")

        temporary_path.replace(
            output_path
        )

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def build_unique_image_records(
    dataset,
):
    records_by_image_id = {}

    for index in range(len(dataset)):
        sample = dataset[index]

        image_id = str(
            sample["image_id"]
        )

        image_path = Path(
            sample["image_path"]
        ).resolve()

        if image_id in records_by_image_id:
            previous_path = (
                records_by_image_id[
                    image_id
                ]["image_path"]
            )

            if image_path != previous_path:
                raise ValueError(
                    "One image ID refers to "
                    "multiple paths: "
                    f"{image_id}"
                )

            continue

        records_by_image_id[
            image_id
        ] = {
            "image_id": image_id,
            "image_path": image_path,
            "first_question_id": str(
                sample["question_id"]
            ),
        }

    return tuple(
        records_by_image_id.values()
    )


def image_id_digest(records):
    joined_ids = "\n".join(
        record["image_id"]
        for record in records
    )

    return sha256(
        joined_ids.encode("utf-8")
    ).hexdigest()


def add_detection_counts(
    class_counts,
    detection_result,
):
    for detection in (
        detection_result.detections
    ):
        class_counts[
            detection.class_name
        ] += 1


def validate_existing_cache(
    cache_path,
    record,
    detector,
    configuration,
):
    image_hash = file_sha256(
        record["image_path"]
    )

    return load_object_detection_cache(
        input_path=cache_path,
        expected_image_id=(
            record["image_id"]
        ),
        expected_image_sha256=image_hash,
        expected_model_name=(
            detector.weights_path.name
        ),
        expected_model_version=(
            detector.ultralytics_version
        ),
        expected_weights_sha256=(
            detector.weights_sha256
        ),
        expected_configuration=(
            configuration
        ),
    )


def main():
    args = parse_args()

    args.manifest = (
        args.manifest
        .expanduser()
        .resolve()
    )

    args.data_root = (
        args.data_root
        .expanduser()
        .resolve()
    )

    args.weights = (
        args.weights
        .expanduser()
        .resolve()
    )

    args.cache_root = (
        args.cache_root
        .expanduser()
        .resolve()
    )

    if args.max_new_images < 0:
        raise ValueError(
            "--max-new-images cannot "
            "be negative."
        )

    if args.progress_every <= 0:
        raise ValueError(
            "--progress-every must "
            "be positive."
        )

    dataset = GQADataset(
        manifest_path=args.manifest,
        data_root=args.data_root,
        verify_images=True,
    )

    records = build_unique_image_records(
        dataset
    )

    print("GQA YOLO detection cache:")
    print("  Manifest:", args.manifest)
    print("  Samples:", len(dataset))
    print("  Unique images:", len(records))
    print("  Weights:", args.weights)
    print("  Cache root:", args.cache_root)
    print(
        "  Inference size:",
        args.inference_size,
    )
    print(
        "  Confidence threshold:",
        args.confidence_threshold,
    )
    print(
        "  IoU threshold:",
        args.iou_threshold,
    )
    print(
        "  Max detections:",
        args.max_detections,
    )
    print("  Device:", args.device)
    print(
        "  Maximum new images:",
        (
            "unlimited"
            if args.max_new_images == 0
            else args.max_new_images
        ),
    )
    print("  Force:", args.force)

    print("\nLoading YOLO11 detector...")

    detector = YOLO11ObjectDetector(
        weights_path=args.weights,
        inference_size=(
            args.inference_size
        ),
        confidence_threshold=(
            args.confidence_threshold
        ),
        iou_threshold=(
            args.iou_threshold
        ),
        max_detections=(
            args.max_detections
        ),
        device=args.device,
    )

    configuration = {
        "inference_size": (
            args.inference_size
        ),
        "confidence_threshold": (
            args.confidence_threshold
        ),
        "iou_threshold": (
            args.iou_threshold
        ),
        "max_detections": (
            args.max_detections
        ),
        "device": args.device,
    }

    print(
        "  Ultralytics version:",
        detector.ultralytics_version,
    )
    print(
        "  Weights SHA256:",
        detector.weights_sha256,
    )

    valid_results = {}
    pending_records = []
    invalid_records = []
    class_counts = Counter()

    print("\nScanning existing caches...")

    for record in records:
        cache_path = detection_cache_path(
            cache_root=args.cache_root,
            image_id=record["image_id"],
        )

        if args.force or not cache_path.is_file():
            pending_records.append(record)
            continue

        try:
            detection_result = (
                validate_existing_cache(
                    cache_path=cache_path,
                    record=record,
                    detector=detector,
                    configuration=(
                        configuration
                    ),
                )
            )

        except (
            OSError,
            ValueError,
            TypeError,
        ) as error:
            invalid_records.append(
                {
                    "record": record,
                    "error": str(error),
                }
            )
            pending_records.append(record)
            continue

        valid_results[
            record["image_id"]
        ] = detection_result

        add_detection_counts(
            class_counts,
            detection_result,
        )

    reused_count = len(valid_results)

    print(
        "  Reusable caches:",
        reused_count,
    )
    print(
        "  Missing or forced caches:",
        len(pending_records),
    )
    print(
        "  Invalid existing caches:",
        len(invalid_records),
    )

    if invalid_records:
        print(
            "\nFirst invalid cache diagnostics:"
        )

        for invalid in invalid_records[:5]:
            print(
                " ",
                invalid["record"]["image_id"],
                "-",
                invalid["error"],
            )

    if args.max_new_images > 0:
        records_to_process = (
            pending_records[
                :args.max_new_images
            ]
        )
    else:
        records_to_process = (
            pending_records
        )

    run_start = time.perf_counter()
    inference_times = []
    newly_created = 0

    for run_index, record in enumerate(
        records_to_process,
        start=1,
    ):
        image_start = time.perf_counter()

        detection_result = detector.detect(
            image_path=(
                record["image_path"]
            ),
            image_id=record["image_id"],
        )

        output_path = detection_cache_path(
            cache_root=args.cache_root,
            image_id=record["image_id"],
        )

        save_object_detection_cache(
            output_path=output_path,
            detection_result=(
                detection_result
            ),
        )

        reloaded_result = (
            validate_existing_cache(
                cache_path=output_path,
                record=record,
                detector=detector,
                configuration=(
                    configuration
                ),
            )
        )

        valid_results[
            record["image_id"]
        ] = reloaded_result

        add_detection_counts(
            class_counts,
            reloaded_result,
        )

        newly_created += 1

        elapsed_for_image = (
            time.perf_counter()
            - image_start
        )

        inference_times.append(
            elapsed_for_image
        )

        completed_count = len(
            valid_results
        )

        should_print = (
            run_index == 1
            or run_index
            % args.progress_every
            == 0
            or run_index
            == len(records_to_process)
        )

        if should_print:
            average_time = (
                sum(inference_times)
                / len(inference_times)
            )

            remaining_count = (
                len(records)
                - completed_count
            )

            estimated_remaining = (
                average_time
                * remaining_count
            )

            print(
                f"  [{completed_count:4d}/"
                f"{len(records)}] "
                f"image_id="
                f"{record['image_id']}, "
                f"detections="
                f"{len(reloaded_result.detections)}, "
                f"time={elapsed_for_image:.2f}s, "
                f"remaining≈"
                f"{estimated_remaining / 60:.1f}m"
            )

    completed_count = len(
        valid_results
    )

    remaining_count = (
        len(records)
        - completed_count
    )

    status = (
        "complete"
        if remaining_count == 0
        else "partial"
    )

    total_detection_count = sum(
        len(result.detections)
        for result in valid_results.values()
    )

    cache_size_bytes = sum(
        detection_cache_path(
            cache_root=args.cache_root,
            image_id=image_id,
        ).stat().st_size
        for image_id in valid_results
    )

    manifest_output_path = (
        args.cache_root
        / "manifests"
        / f"{args.manifest.stem}.json"
    )

    metadata = {
        "schema_version": 1,
        "status": status,
        "updated_at_utc": (
            utc_timestamp()
        ),
        "dataset_manifest": str(
            args.manifest
        ),
        "dataset_samples": len(dataset),
        "unique_images": len(records),
        "image_id_sha256": (
            image_id_digest(records)
        ),
        "image_ids": [
            record["image_id"]
            for record in records
        ],
        "cache_root": str(
            args.cache_root
        ),
        "entries_directory": str(
            args.cache_root / "entries"
        ),
        "detector": {
            "detector_name": (
                "ultralytics_yolo"
            ),
            "model_name": (
                detector.weights_path.name
            ),
            "model_version": (
                detector.ultralytics_version
            ),
            "weights_path": str(
                detector.weights_path
            ),
            "weights_sha256": (
                detector.weights_sha256
            ),
        },
        "configuration": (
            configuration
        ),
        "completed_images": (
            completed_count
        ),
        "remaining_images": (
            remaining_count
        ),
        "reused_images": reused_count,
        "newly_created_images": (
            newly_created
        ),
        "invalid_existing_images": (
            len(invalid_records)
        ),
        "total_detections": (
            total_detection_count
        ),
        "class_counts": dict(
            sorted(
                class_counts.items()
            )
        ),
        "cache_size_bytes": (
            cache_size_bytes
        ),
        "elapsed_seconds": (
            time.perf_counter()
            - run_start
        ),
    }

    write_json_atomic(
        output_path=(
            manifest_output_path
        ),
        payload=metadata,
    )

    print(
        "\nGQA YOLO cache summary:"
    )
    print(
        "  Status:",
        status,
    )
    print(
        "  Completed:",
        f"{completed_count}/{len(records)}",
    )
    print(
        "  Reused:",
        reused_count,
    )
    print(
        "  Newly created:",
        newly_created,
    )
    print(
        "  Remaining:",
        remaining_count,
    )
    print(
        "  Total detections:",
        total_detection_count,
    )
    print(
        "  Total cache size:",
        f"{cache_size_bytes / (1024 ** 2):.3f} MiB",
    )
    print(
        "  Metadata:",
        manifest_output_path,
    )

    most_common_classes = (
        class_counts.most_common(15)
    )

    print("\nMost common classes:")

    for class_name, count in (
        most_common_classes
    ):
        print(
            f"  {class_name:<24} "
            f"{count}"
        )

    if status == "complete":
        print(
            "\nGQA YOLO detection "
            "cache generation: PASS"
        )
    else:
        print(
            "\nGQA YOLO detection "
            "cache generation: PARTIAL"
        )
        print(
            "Run the same command again "
            "to resume."
        )


if __name__ == "__main__":
    main()
