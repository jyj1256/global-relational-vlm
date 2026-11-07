import argparse
import json
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from vlm.data.gqa_dataset import (
    GQADataset,
    normalize_gqa_answer,
)
from vlm.scripts.test_backbone import (
    load_image,
)
from vlm.utils.first_pass_cache import (
    build_cache_identity,
    load_first_pass_cache,
)
from vlm.utils.object_detection_cache import (
    detection_cache_path,
    load_object_detection_cache,
)
from vlm.utils.training_checkpoint import (
    load_training_checkpoint,
)
from vlm.utils.training_model_setup import (
    load_training_model,
)
from vlm.utils.training_runtime import (
    append_jsonl,
    file_sha256,
    load_cache_manifest,
    load_json,
    write_json_atomic,
)


DEFAULT_GQA_ROOT = (
    PROJECT_ROOT / "datasets/gqa"
)
DEFAULT_MANIFEST = (
    DEFAULT_GQA_ROOT
    / "subsets/val_debug_100.jsonl"
)
DEFAULT_CACHE_ROOT = (
    PROJECT_ROOT
    / "vlm/outputs/first_pass_cache/"
    "nf4_qlora_maxnum1_val_debug100"
)
DEFAULT_DETECTION_CACHE_ROOT = (
    PROJECT_ROOT
    / "vlm/outputs/object_detections/"
    "yolo11s_coco"
)
DEFAULT_DETECTION_CACHE_MANIFEST = (
    DEFAULT_DETECTION_CACHE_ROOT
    / "manifests/val_debug_100.json"
)
DEFAULT_TRAINING_ROOT = (
    PROJECT_ROOT
    / "vlm/outputs/training/"
    "debug100_full_epoch1"
)
DEFAULT_CHECKPOINT = (
    DEFAULT_TRAINING_ROOT
    / "checkpoints/step_000013.pt"
)
DEFAULT_RUN_CONFIG = (
    DEFAULT_TRAINING_ROOT
    / "run_config.json"
)
DEFAULT_STATISTICS = (
    PROJECT_ROOT
    / "vlm/configs/"
    "gqa_descriptor_statistics_"
    "nf4_qlora_5000_seed2026.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "vlm/outputs/predictions/"
    "val_debug100_step13_raw_greedy"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate cached Global-Relational "
            "VLM generation on GQA."
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
        default=DEFAULT_GQA_ROOT,
    )
    parser.add_argument(
        "--first-pass-cache",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
    )
    parser.add_argument(
        "--graph-version",
        type=str,
        choices=("v0", "v1"),
        default="v0",
    )
    parser.add_argument(
        "--detection-cache-root",
        type=Path,
        default=(
            DEFAULT_DETECTION_CACHE_ROOT
        ),
    )
    parser.add_argument(
        "--detection-cache-manifest",
        type=Path,
        default=(
            DEFAULT_DETECTION_CACHE_MANIFEST
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--run-config",
        type=Path,
        default=DEFAULT_RUN_CONFIG,
    )
    parser.add_argument(
        "--statistics",
        type=Path,
        default=DEFAULT_STATISTICS,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="OpenGVLab/InternVL3-8B",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "pretrained/huggingface"
        ),
    )
    parser.add_argument(
        "--offload-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "pretrained/offload"
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--max-new-samples",
        type=int,
        default=0,
        help=(
            "Process at most this many new "
            "samples. Zero processes all."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
    )

    return parser.parse_args()


def load_detection_manifest(
    cache_root,
    metadata_path,
    dataset,
):
    if not cache_root.is_dir():
        raise FileNotFoundError(
            cache_root
        )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            metadata_path
        )

    metadata = load_json(
        metadata_path
    )

    dataset_image_ids = sorted({
        str(
            dataset[index][
                "image_id"
            ]
        )
        for index in range(len(dataset))
    })

    cached_image_ids = sorted(
        str(image_id)
        for image_id in metadata.get(
            "image_ids",
            [],
        )
    )

    if metadata.get("status") != "complete":
        raise RuntimeError(
            "Object Detection cache is "
            "not complete."
        )

    if cached_image_ids != (
        dataset_image_ids
    ):
        raise ValueError(
            "Object Detection cache image "
            "IDs do not match the dataset."
        )

    if int(
        metadata.get(
            "unique_images",
            -1,
        )
    ) != len(dataset_image_ids):
        raise ValueError(
            "Object Detection cache unique "
            "image count does not match."
        )

    if int(
        metadata.get(
            "completed_images",
            -1,
        )
    ) != len(dataset_image_ids):
        raise ValueError(
            "Object Detection cache completed "
            "image count does not match."
        )

    detector = metadata.get(
        "detector"
    )
    configuration = metadata.get(
        "configuration"
    )

    if not isinstance(detector, dict):
        raise ValueError(
            "Detection metadata does not "
            "contain detector information."
        )

    required_detector_fields = (
        "model_name",
        "model_version",
        "weights_sha256",
    )

    for field in required_detector_fields:
        if not detector.get(field):
            raise ValueError(
                "Detection metadata is missing "
                f"detector field: {field}."
            )

    if not isinstance(
        configuration,
        dict,
    ):
        raise ValueError(
            "Detection metadata does not "
            "contain configuration."
        )

    for image_id in dataset_image_ids:
        entry_path = (
            detection_cache_path(
                cache_root,
                image_id,
            )
        )

        if not entry_path.is_file():
            raise FileNotFoundError(
                entry_path
            )

    return metadata


def load_existing_predictions(
    predictions_path,
    dataset_question_ids,
):
    predictions = {}

    if not predictions_path.is_file():
        return predictions

    with predictions_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)
            question_id = str(
                record["question_id"]
            )

            if question_id not in (
                dataset_question_ids
            ):
                raise ValueError(
                    "Prediction contains an "
                    "unknown question ID at "
                    f"line {line_number}."
                )

            if question_id in predictions:
                raise ValueError(
                    "Duplicate prediction for "
                    f"{question_id}."
                )

            predictions[
                question_id
            ] = record

    return predictions


def grouped_metrics(records, type_name):
    groups = {}

    for record in records:
        group_name = (
            record["types"][type_name]
        )

        if group_name not in groups:
            groups[group_name] = {
                "count": 0,
                "correct": 0,
            }

        groups[group_name]["count"] += 1
        groups[group_name]["correct"] += int(
            record["exact_match"]
        )

    for values in groups.values():
        values["accuracy"] = (
            values["correct"]
            / values["count"]
        )

    return dict(
        sorted(groups.items())
    )


def build_summary(
    records,
    expected_count,
    evaluation_configuration,
):
    correct = sum(
        int(record["exact_match"])
        for record in records
    )

    completed = len(records)

    generation_seconds = sum(
        float(
            record["generation_seconds"]
        )
        for record in records
    )

    return {
        "status": (
            "complete"
            if completed == expected_count
            else "partial"
        ),
        "metric": (
            "conservative_normalized_"
            "exact_match"
        ),
        "completed_count": completed,
        "expected_count": expected_count,
        "correct": correct,
        "accuracy": (
            correct / completed
            if completed > 0
            else 0.0
        ),
        "empty_predictions": sum(
            not record[
                "normalized_prediction"
            ]
            for record in records
        ),
        "average_generation_seconds": (
            generation_seconds
            / completed
            if completed > 0
            else 0.0
        ),
        "structural": grouped_metrics(
            records,
            "structural",
        ),
        "semantic": grouped_metrics(
            records,
            "semantic",
        ),
        "evaluation_configuration": (
            evaluation_configuration
        ),
    }


def main():
    args = parse_args()

    if args.max_new_tokens <= 0:
        raise ValueError(
            "--max-new-tokens must be "
            "positive."
        )

    if args.max_new_samples < 0:
        raise ValueError(
            "--max-new-samples cannot be "
            "negative."
        )

    if args.progress_every <= 0:
        raise ValueError(
            "--progress-every must be "
            "positive."
        )

    for path in (
        args.manifest,
        args.checkpoint,
        args.run_config,
        args.statistics,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    if args.graph_version == "v1":
        if not (
            args
            .detection_cache_root
            .is_dir()
        ):
            raise FileNotFoundError(
                args.detection_cache_root
            )

        if not (
            args
            .detection_cache_manifest
            .is_file()
        ):
            raise FileNotFoundError(
                args.detection_cache_manifest
            )

    dataset = GQADataset(
        manifest_path=args.manifest,
        data_root=args.data_root,
        verify_images=True,
    )

    (
        cache_metadata,
        cache_record_by_id,
        cache_metadata_path,
    ) = load_cache_manifest(
        cache_root=args.first_pass_cache,
        manifest_path=args.manifest,
        dataset=dataset,
    )

    training_configuration = (
        load_json(args.run_config)
    )

    checkpoint_graph_version = str(
        training_configuration.get(
            "graph_version",
            "v0",
        )
    )

    if checkpoint_graph_version != (
        args.graph_version
    ):
        raise ValueError(
            "Requested graph version does "
            "not match the training run: "
            f"requested={args.graph_version}, "
            f"checkpoint="
            f"{checkpoint_graph_version}."
        )

    detection_metadata = None

    if args.graph_version == "v1":
        detection_metadata = (
            load_detection_manifest(
                cache_root=(
                    args
                    .detection_cache_root
                ),
                metadata_path=(
                    args
                    .detection_cache_manifest
                ),
                dataset=dataset,
            )
        )

    generation_configuration = {
        "max_new_tokens": (
            args.max_new_tokens
        ),
        "do_sample": False,
        "num_beams": 1,
    }

    evaluation_configuration = {
        "protocol": (
            "raw_question_greedy_v1"
        ),
        "model": args.model,
        "manifest": str(
            args.manifest.resolve()
        ),
        "manifest_sha256": (
            file_sha256(args.manifest)
        ),
        "checkpoint": str(
            args.checkpoint.resolve()
        ),
        "checkpoint_sha256": (
            file_sha256(args.checkpoint)
        ),
        "run_config": str(
            args.run_config.resolve()
        ),
        "run_config_sha256": (
            file_sha256(args.run_config)
        ),
        "first_pass_cache": str(
            args
            .first_pass_cache
            .resolve()
        ),
        "cache_metadata_sha256": (
            file_sha256(
                cache_metadata_path
            )
        ),
        "first_pass_configuration": (
            cache_metadata[
                "configuration"
            ]
        ),
        "generation": (
            generation_configuration
        ),
        "normalization": (
            "NFKC + trim + lowercase + "
            "collapse whitespace"
        ),
        "primary_metric": (
            "strict normalized exact match"
        ),
    }

    if args.graph_version == "v1":
        evaluation_configuration[
            "graph_version"
        ] = "v1"

        evaluation_configuration[
            "object_detection_cache"
        ] = {
            "cache_root": str(
                args
                .detection_cache_root
                .resolve()
            ),
            "metadata_path": str(
                args
                .detection_cache_manifest
                .resolve()
            ),
            "metadata_sha256": (
                file_sha256(
                    args
                    .detection_cache_manifest
                )
            ),
            "unique_images": (
                detection_metadata[
                    "unique_images"
                ]
            ),
            "image_id_sha256": (
                detection_metadata.get(
                    "image_id_sha256"
                )
            ),
            "detector": (
                detection_metadata[
                    "detector"
                ]
            ),
            "configuration": (
                detection_metadata[
                    "configuration"
                ]
            ),
        }

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation_config_path = (
        args.output_dir
        / "evaluation_config.json"
    )
    predictions_path = (
        args.output_dir
        / "predictions.jsonl"
    )
    summary_path = (
        args.output_dir
        / "summary.json"
    )

    if evaluation_config_path.is_file():
        existing_configuration = (
            load_json(
                evaluation_config_path
            )
        )

        if existing_configuration != (
            evaluation_configuration
        ):
            raise ValueError(
                "Existing evaluation "
                "configuration does not match."
            )
    else:
        if predictions_path.is_file():
            raise RuntimeError(
                "Predictions exist without an "
                "evaluation configuration."
            )

        write_json_atomic(
            evaluation_config_path,
            evaluation_configuration,
        )

    dataset_question_ids = {
        str(
            dataset[index][
                "question_id"
            ]
        )
        for index in range(len(dataset))
    }

    predictions = (
        load_existing_predictions(
            predictions_path,
            dataset_question_ids,
        )
    )

    missing_samples = [
        dataset[index]
        for index in range(len(dataset))
        if str(
            dataset[index][
                "question_id"
            ]
        )
        not in predictions
    ]

    if args.max_new_samples > 0:
        samples_to_process = (
            missing_samples[
                :args.max_new_samples
            ]
        )
    else:
        samples_to_process = (
            missing_samples
        )

    print("GQA evaluation:")
    print("  Samples:", len(dataset))
    print(
        "  Existing predictions:",
        len(predictions),
    )
    print(
        "  Missing predictions:",
        len(missing_samples),
    )
    print(
        "  New predictions this run:",
        len(samples_to_process),
    )
    print(
        "  Protocol:",
        evaluation_configuration[
            "protocol"
        ],
    )
    print(
        "  Graph version:",
        args.graph_version,
    )

    if args.graph_version == "v1":
        print(
            "  Detection metadata:",
            args.detection_cache_manifest,
        )

    if samples_to_process:
        components = load_training_model(
            model_name=args.model,
            model_cache_dir=(
                args.model_cache_dir
            ),
            offload_dir=(
                args.offload_dir
            ),
            statistics_path=(
                args.statistics
            ),
            gpu_memory="10GiB",
            cpu_memory="48GiB",
            graph_version=(
                args.graph_version
            ),
        )

        integrated_model = components[
            "integrated_model"
        ]
        tokenizer = components[
            "tokenizer"
        ]
        vision_device = components[
            "vision_device"
        ]

        checkpoint_information = (
            load_training_checkpoint(
                input_path=args.checkpoint,
                integrated_model=(
                    integrated_model
                ),
                optimizer=None,
                scheduler=None,
                expected_configuration=(
                    training_configuration
                ),
                adapter_name=(
                    "global_relational"
                ),
                restore_rng=False,
            )
        )

        integrated_model.eval()

        extraction_calls = {
            "count": 0,
        }

        def forbidden_first_pass(
            *unused_args,
            **unused_kwargs,
        ):
            extraction_calls["count"] += 1

            raise RuntimeError(
                "First Pass was recomputed "
                "during cached evaluation."
            )

        integrated_model.extract_first_pass = (
            forbidden_first_pass
        )

        generation_configuration[
            "pad_token_id"
        ] = tokenizer.eos_token_id

        start_time = time.perf_counter()

        for new_index, sample in enumerate(
            samples_to_process,
            start=1,
        ):
            question_id = str(
                sample["question_id"]
            )

            cache_record = (
                cache_record_by_id[
                    question_id
                ]
            )
            cache_path = (
                args.first_pass_cache
                / cache_record[
                    "cache_file"
                ]
            )

            identity = (
                build_cache_identity(
                    question_id=(
                        question_id
                    ),
                    image_id=(
                        sample["image_id"]
                    ),
                    question=(
                        sample["question"]
                    ),
                )
            )

            (
                first_pass,
                _,
            ) = load_first_pass_cache(
                input_path=cache_path,
                expected_identity=(
                    identity
                ),
                expected_configuration=(
                    cache_metadata[
                        "configuration"
                    ]
                ),
                tile_size=448,
            )

            (
                original_image,
                pixel_values,
                tile_grid,
            ) = load_image(
                image_path=Path(
                    sample["image_path"]
                ),
                image_size=448,
                max_num=1,
            )

            pixel_values = pixel_values.to(
                device=vision_device,
                dtype=torch.bfloat16,
            )

            detections = None

            if args.graph_version == "v1":
                image_id = str(
                    sample["image_id"]
                )
                detector_metadata = (
                    detection_metadata[
                        "detector"
                    ]
                )

                detection_result = (
                    load_object_detection_cache(
                        input_path=(
                            detection_cache_path(
                                args
                                .detection_cache_root,
                                image_id,
                            )
                        ),
                        expected_image_id=(
                            image_id
                        ),
                        expected_image_sha256=(
                            file_sha256(
                                Path(
                                    sample[
                                        "image_path"
                                    ]
                                )
                            )
                        ),
                        expected_model_name=(
                            detector_metadata[
                                "model_name"
                            ]
                        ),
                        expected_model_version=(
                            detector_metadata[
                                "model_version"
                            ]
                        ),
                        expected_weights_sha256=(
                            detector_metadata[
                                "weights_sha256"
                            ]
                        ),
                        expected_configuration=(
                            detection_metadata[
                                "configuration"
                            ]
                        ),
                    )
                )

                if tuple(
                    detection_result.image_size
                ) != tuple(
                    original_image.size
                ):
                    raise ValueError(
                        "Detection image size "
                        "does not match the "
                        "evaluation image."
                    )

                detections = (
                    detection_result.detections
                )

            sample_start = (
                time.perf_counter()
            )

            generation_output = (
                integrated_model.generate(
                    pixel_values=(
                        pixel_values
                    ),
                    question=(
                        sample["question"]
                    ),
                    tile_grid=tile_grid,
                    image_size=(
                        original_image.size
                    ),
                    generation_config=(
                        generation_configuration
                    ),
                    first_pass=first_pass,
                    detections=detections,
                )
            )

            generation_seconds = (
                time.perf_counter()
                - sample_start
            )

            diagnostics = (
                generation_output
                .second_pass_diagnostics
            )

            if (
                diagnostics.get(
                    "first_pass_cached"
                )
                is not True
            ):
                raise RuntimeError(
                    "Cached First Pass was "
                    "not used."
                )

            if (
                diagnostics.get(
                    "adapter_enabled"
                )
                is not True
            ):
                raise RuntimeError(
                    "QLoRA Adapter was not "
                    "enabled."
                )

            if (
                diagnostics.get(
                    "graph_version"
                )
                != args.graph_version
            ):
                raise RuntimeError(
                    "Generation used an "
                    "unexpected graph version."
                )

            if (
                args.graph_version == "v1"
                and int(
                    diagnostics.get(
                        "detection_count",
                        -1,
                    )
                )
                != len(detections)
            ):
                raise RuntimeError(
                    "Generation detection count "
                    "does not match the cache."
                )

            raw_prediction = (
                generation_output.response
            )
            normalized_prediction = (
                normalize_gqa_answer(
                    raw_prediction
                )
            )
            normalized_reference = (
                normalize_gqa_answer(
                    sample["answer"]
                )
            )

            record = {
                "question_id": question_id,
                "image_id": str(
                    sample["image_id"]
                ),
                "question": (
                    sample["question"]
                ),
                "reference_answer": (
                    sample["answer"]
                ),
                "normalized_reference": (
                    normalized_reference
                ),
                "prediction": (
                    raw_prediction
                ),
                "normalized_prediction": (
                    normalized_prediction
                ),
                "exact_match": (
                    normalized_prediction
                    == normalized_reference
                ),
                "types": sample["types"],
                "generation_seconds": (
                    generation_seconds
                ),
                "input_order": (
                    diagnostics.get(
                        "input_order"
                    )
                ),
                "global_token_norm": float(
                    generation_output
                    .global_output
                    .global_token
                    .float()
                    .norm()
                    .item()
                ),
                "graph_token_norm": float(
                    generation_output
                    .graph_output
                    .graph_token
                    .float()
                    .norm()
                    .item()
                ),
                "checkpoint_optimizer_step": (
                    checkpoint_information[
                        "progress"
                    ]["optimizer_step"]
                ),
            }

            if args.graph_version == "v1":
                record["graph_version"] = "v1"
                record["detection_count"] = (
                    len(detections)
                )
                record["graph_node_count"] = (
                    len(
                        generation_output
                        .graph_output
                        .hybrid_output
                        .nodes
                    )
                )
                record[
                    "graph_edge_count"
                ] = (
                    generation_output
                    .graph_output
                    .graph_output
                    .undirected_edge_count
                )

            append_jsonl(
                predictions_path,
                record,
            )

            predictions[
                question_id
            ] = record

            if (
                new_index
                % args.progress_every
                == 0
                or new_index
                == len(samples_to_process)
            ):
                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                print(
                    f"  [{len(predictions):3d}/"
                    f"{len(dataset)}] "
                    f"question_id="
                    f"{question_id}, "
                    f"exact="
                    f"{record['exact_match']}, "
                    f"time="
                    f"{generation_seconds:.2f}s, "
                    f"elapsed="
                    f"{elapsed / 60:.1f}m"
                )

            del generation_output
            del first_pass
            del pixel_values
            del original_image

            if (
                torch.cuda.is_available()
                and new_index % 20 == 0
            ):
                torch.cuda.empty_cache()

        if extraction_calls["count"] != 0:
            raise RuntimeError(
                "First Pass extraction was "
                "unexpectedly called."
            )

    ordered_records = [
        predictions[
            str(
                dataset[index][
                    "question_id"
                ]
            )
        ]
        for index in range(len(dataset))
        if str(
            dataset[index][
                "question_id"
            ]
        )
        in predictions
    ]

    summary = build_summary(
        records=ordered_records,
        expected_count=len(dataset),
        evaluation_configuration=(
            evaluation_configuration
        ),
    )

    write_json_atomic(
        summary_path,
        summary,
    )

    print("\nEvaluation summary:")
    print(
        "  Status:",
        summary["status"],
    )
    print(
        "  Completed:",
        summary["completed_count"],
        "/",
        summary["expected_count"],
    )
    print(
        "  Exact Match:",
        (
            f"{summary['correct']}/"
            f"{summary['completed_count']} "
            f"({summary['accuracy']:.4f})"
        ),
    )
    print(
        "  Empty predictions:",
        summary[
            "empty_predictions"
        ],
    )
    print(
        "  Average generation time:",
        f"{summary['average_generation_seconds']:.3f}s",
    )

    print("\nStructural accuracy:")

    for name, values in (
        summary["structural"].items()
    ):
        print(
            f"  {name:<10} "
            f"{values['correct']}/"
            f"{values['count']} "
            f"({values['accuracy']:.4f})"
        )

    if summary["status"] == "complete":
        print(
            "\nGQA evaluation pipeline: PASS"
        )
    else:
        print(
            "\nGQA evaluation pipeline: PARTIAL"
        )
        print(
            "Run the same command again to "
            "resume."
        )

    print(
        "Note: this Debug-100 result is not "
        "a paper-level performance claim."
    )


if __name__ == "__main__":
    main()