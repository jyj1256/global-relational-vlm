import json
from pathlib import Path
from typing import Mapping, Optional

from vlm.detectors.detection_types import (
    ObjectDetectionResult,
)


CONFIGURATION_FIELDS = {
    "inference_size",
    "confidence_threshold",
    "iou_threshold",
    "max_detections",
    "device",
}


def detection_cache_path(
    cache_root,
    image_id,
):
    cache_root = Path(cache_root)
    image_id = str(image_id).strip()

    if not image_id:
        raise ValueError(
            "image_id must not be empty."
        )

    if (
        Path(image_id).name != image_id
        or "/" in image_id
        or "\\" in image_id
    ):
        raise ValueError(
            "image_id must not contain "
            "a directory component."
        )

    return (
        cache_root
        / "entries"
        / f"{image_id}.json"
    )


def save_object_detection_cache(
    output_path,
    detection_result,
):
    if not isinstance(
        detection_result,
        ObjectDetectionResult,
    ):
        raise TypeError(
            "detection_result must be an "
            "ObjectDetectionResult."
        )

    detection_result.validate()

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

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return output_path


def _load_payload(
    input_path,
):
    input_path = Path(input_path)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Detection cache not found: "
            f"{input_path}"
        )

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(
            "Detection cache root must "
            "be a dictionary."
        )

    return payload


def _matches_expected_value(
    actual,
    expected,
):
    if isinstance(expected, float):
        return (
            abs(
                float(actual)
                - float(expected)
            )
            <= 1e-12
        )

    return actual == expected


def _validate_expected_identity(
    detection_result,
    expected_image_id=None,
    expected_image_sha256=None,
    expected_model_name=None,
    expected_model_version=None,
    expected_weights_sha256=None,
):
    identity_checks = (
        (
            "image_id",
            detection_result.image_id,
            expected_image_id,
        ),
        (
            "image_sha256",
            detection_result.image_sha256,
            expected_image_sha256,
        ),
        (
            "model_name",
            detection_result.model_name,
            expected_model_name,
        ),
        (
            "model_version",
            detection_result.model_version,
            expected_model_version,
        ),
        (
            "weights_sha256",
            detection_result.weights_sha256,
            expected_weights_sha256,
        ),
    )

    for name, actual, expected in (
        identity_checks
    ):
        if expected is None:
            continue

        normalized_expected = str(
            expected
        )

        if name.endswith("sha256"):
            normalized_expected = (
                normalized_expected.lower()
            )

        if actual != normalized_expected:
            raise ValueError(
                f"Detection cache {name} "
                f"mismatch: expected "
                f"{normalized_expected!r}, "
                f"found {actual!r}."
            )


def _validate_expected_configuration(
    detection_result,
    expected_configuration,
):
    if expected_configuration is None:
        return

    if not isinstance(
        expected_configuration,
        Mapping,
    ):
        raise TypeError(
            "expected_configuration must "
            "be a mapping."
        )

    unknown_fields = (
        set(expected_configuration)
        - CONFIGURATION_FIELDS
    )

    if unknown_fields:
        raise ValueError(
            "Unknown expected configuration "
            f"fields: {sorted(unknown_fields)}"
        )

    for name, expected in (
        expected_configuration.items()
    ):
        actual = getattr(
            detection_result,
            name,
        )

        if not _matches_expected_value(
            actual,
            expected,
        ):
            raise ValueError(
                "Detection cache configuration "
                f"mismatch for {name}: "
                f"expected {expected!r}, "
                f"found {actual!r}."
            )


def load_object_detection_cache(
    input_path,
    expected_image_id: Optional[str] = None,
    expected_image_sha256: Optional[str] = None,
    expected_model_name: Optional[str] = None,
    expected_model_version: Optional[str] = None,
    expected_weights_sha256: Optional[str] = None,
    expected_configuration: Optional[
        Mapping
    ] = None,
):
    payload = _load_payload(
        input_path
    )

    try:
        detection_result = (
            ObjectDetectionResult.from_dict(
                payload
            )
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            "Invalid object detection cache: "
            f"{input_path}"
        ) from error

    _validate_expected_identity(
        detection_result=(
            detection_result
        ),
        expected_image_id=(
            expected_image_id
        ),
        expected_image_sha256=(
            expected_image_sha256
        ),
        expected_model_name=(
            expected_model_name
        ),
        expected_model_version=(
            expected_model_version
        ),
        expected_weights_sha256=(
            expected_weights_sha256
        ),
    )

    _validate_expected_configuration(
        detection_result=(
            detection_result
        ),
        expected_configuration=(
            expected_configuration
        ),
    )

    return detection_result