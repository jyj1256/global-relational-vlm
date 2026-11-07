import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from vlm.data.gqa_dataset import (
    GQADataset,
    build_gqa_dataloader,
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
    collect_trainable_parameters,
    create_optimizer_and_scheduler,
    file_sha256,
    gradient_norm,
    load_cache_manifest,
    save_runner_checkpoint,
    seed_everything,
    write_json_atomic,
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
DEFAULT_FIRST_PASS_CACHE = (
    PROJECT_ROOT
    / "vlm/outputs/first_pass_cache/"
    "nf4_qlora_maxnum1_train_seed2026"
)
DEFAULT_DETECTION_CACHE = (
    PROJECT_ROOT
    / "vlm/outputs/object_detections/"
    "yolo11s_coco"
)

DEFAULT_STATISTICS = (
    PROJECT_ROOT
    / "vlm/configs/"
    "gqa_descriptor_statistics_"
    "nf4_qlora_5000_seed2026.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "vlm/outputs/training/"
    "debug100_cached_qlora"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the cached Two-Pass "
            "Global-Relational VLM."
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
        "--first-pass-cache",
        type=Path,
        default=DEFAULT_FIRST_PASS_CACHE,
    )
    parser.add_argument(
        "--statistics",
        type=Path,
        default=DEFAULT_STATISTICS,
    )
    parser.add_argument(
        "--graph-version",
        type=str,
        choices=(
            "v0",
            "v1",
        ),
        default="v0",
    )
    parser.add_argument(
        "--detection-cache-root",
        type=Path,
        default=DEFAULT_DETECTION_CACHE,
    )
    parser.add_argument(
        "--detection-cache-manifest",
        type=Path,
        default=None,
        help=(
            "Optional explicit YOLO cache "
            "manifest. By default it is "
            "derived from --manifest."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
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
        "--gpu-memory",
        type=str,
        default="10GiB",
    )
    parser.add_argument(
        "--cpu-memory",
        type=str,
        default="48GiB",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--auxiliary-lr",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--qlora-lr",
        type=float,
        default=5e-5,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-optimizer-steps",
        type=int,
        default=0,
        help=(
            "Runtime stop limit for smoke and "
            "Resume tests. Zero runs all steps."
        ),
    )

    return parser.parse_args()


def validate_args(args):
    for name in (
        "epochs",
        "gradient_accumulation_steps",
        "checkpoint_every",
        "log_every",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(
                f"--{name.replace('_', '-')} "
                "must be positive."
            )

    if args.max_optimizer_steps < 0:
        raise ValueError(
            "--max-optimizer-steps cannot "
            "be negative."
        )

    if args.auxiliary_lr <= 0:
        raise ValueError(
            "--auxiliary-lr must be positive."
        )

    if args.qlora_lr <= 0:
        raise ValueError(
            "--qlora-lr must be positive."
        )

    if args.weight_decay < 0:
        raise ValueError(
            "--weight-decay cannot be negative."
        )

    if not 0 <= args.warmup_ratio < 1:
        raise ValueError(
            "--warmup-ratio must be in [0, 1)."
        )

    if args.max_grad_norm <= 0:
        raise ValueError(
            "--max-grad-norm must be positive."
        )

    for path in (
        args.manifest,
        args.statistics,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    for path in (
        args.data_root,
        args.first_pass_cache,
    ):
        if not path.is_dir():
            raise FileNotFoundError(path)

    if (
        args.resume is not None
        and not args.resume.is_file()
    ):
        raise FileNotFoundError(
            args.resume
        )

    if args.graph_version == "v1":
        if not (
            args.detection_cache_root
            .is_dir()
        ):
            raise FileNotFoundError(
                args.detection_cache_root
            )

        if (
            args.detection_cache_manifest
            is not None
            and not (
                args
                .detection_cache_manifest
                .is_file()
            )
        ):
            raise FileNotFoundError(
                args.detection_cache_manifest
            )


def load_detection_manifest(
    args,
    dataset,
):
    if args.detection_cache_manifest is None:
        metadata_path = (
            args.detection_cache_root
            / "manifests"
            / f"{args.manifest.stem}.json"
        )
    else:
        metadata_path = (
            args.detection_cache_manifest
        )

    metadata_path = (
        metadata_path
        .expanduser()
        .resolve()
    )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            metadata_path
        )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    if metadata.get("status") != "complete":
        raise RuntimeError(
            "Object detection cache manifest "
            "is not complete."
        )

    ordered_image_ids = list(
        dict.fromkeys(
            str(dataset[index]["image_id"])
            for index in range(
                len(dataset)
            )
        )
    )

    if metadata.get(
        "dataset_samples"
    ) != len(dataset):
        raise RuntimeError(
            "Detection cache sample count "
            "does not match the dataset."
        )

    if metadata.get(
        "unique_images"
    ) != len(ordered_image_ids):
        raise RuntimeError(
            "Detection cache unique-image "
            "count does not match."
        )

    if metadata.get(
        "image_ids"
    ) != ordered_image_ids:
        raise RuntimeError(
            "Detection cache image IDs "
            "do not match the dataset."
        )

    expected_manifest = str(
        args.manifest
        .expanduser()
        .resolve()
    )

    if metadata.get(
        "dataset_manifest"
    ) != expected_manifest:
        raise RuntimeError(
            "Detection cache dataset "
            "manifest does not match."
        )

    if metadata.get(
        "completed_images"
    ) != len(ordered_image_ids):
        raise RuntimeError(
            "Detection cache is missing "
            "completed images."
        )

    required_detector_fields = {
        "model_name",
        "model_version",
        "weights_sha256",
    }

    detector_metadata = metadata.get(
        "detector",
        {},
    )

    missing_detector_fields = (
        required_detector_fields
        - set(detector_metadata)
    )

    if missing_detector_fields:
        raise RuntimeError(
            "Detection manifest is missing "
            "detector fields: "
            f"{sorted(missing_detector_fields)}"
        )

    required_configuration_fields = {
        "inference_size",
        "confidence_threshold",
        "iou_threshold",
        "max_detections",
        "device",
    }

    detection_configuration = (
        metadata.get(
            "configuration",
            {},
        )
    )

    missing_configuration_fields = (
        required_configuration_fields
        - set(detection_configuration)
    )

    if missing_configuration_fields:
        raise RuntimeError(
            "Detection manifest is missing "
            "configuration fields: "
            f"{sorted(missing_configuration_fields)}"
        )

    missing_entries = []

    for image_id in ordered_image_ids:
        entry_path = detection_cache_path(
            cache_root=(
                args.detection_cache_root
            ),
            image_id=image_id,
        )

        if not entry_path.is_file():
            missing_entries.append(
                str(entry_path)
            )

            if len(missing_entries) >= 10:
                break

    if missing_entries:
        raise FileNotFoundError(
            "Missing object detection "
            "entries:\n"
            + "\n".join(
                missing_entries
            )
        )

    return metadata, metadata_path


def initial_progress():
    return {
        "epoch": 0,
        "next_batch_index": 0,
        "global_step": 0,
        "optimizer_step": 0,
        "samples_seen": 0,
        "accumulation_remainder": 0,
    }


def initial_metrics():
    return {
        "total_loss_sum": 0.0,
        "total_sample_count": 0,
        "total_token_correct": 0,
        "total_token_count": 0,
        "epoch_loss_sum": 0.0,
        "epoch_sample_count": 0,
        "epoch_token_correct": 0,
        "epoch_token_count": 0,
    }


def safe_ratio(numerator, denominator):
    return (
        numerator / denominator
        if denominator > 0
        else 0.0
    )


def main():
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)

    latest_pointer_path = (
        args.output_dir
        / "latest_checkpoint.json"
    )
    log_path = (
        args.output_dir
        / "train_log.jsonl"
    )

    if (
        args.resume is None
        and (
            latest_pointer_path.exists()
            or log_path.exists()
        )
    ):
        raise RuntimeError(
            "The output directory already "
            "contains a training run. Use "
            "--resume or choose another "
            "--output-dir."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
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

    cache_configuration = (
        cache_metadata["configuration"]
    )

    image_size = int(
        cache_configuration["image_size"]
    )
    max_num = int(
        cache_configuration["max_num"]
    )

    detection_metadata = None
    detection_metadata_path = None
    detection_configuration = None

    if args.graph_version == "v1":
        (
            detection_metadata,
            detection_metadata_path,
        ) = load_detection_manifest(
            args=args,
            dataset=dataset,
        )

        detection_configuration = (
            detection_metadata[
                "configuration"
            ]
        )

    print("Training configuration:")
    print("  Samples:", len(dataset))
    print("  Epochs:", args.epochs)
    print("  Micro batch size: 1")
    print(
        "  Gradient accumulation:",
        args.gradient_accumulation_steps,
    )
    print(
        "  Auxiliary LR:",
        args.auxiliary_lr,
    )
    print(
        "  QLoRA LR:",
        args.qlora_lr,
    )
    print(
        "  Graph version:",
        args.graph_version,
    )
    print(
        "  Cache metadata:",
        cache_metadata_path,
    )

    if detection_metadata_path is not None:
        print(
            "  Detection metadata:",
            detection_metadata_path,
        )
    print("  Output:", args.output_dir)

    components = load_training_model(
        model_name=args.model,
        model_cache_dir=(
            args.model_cache_dir
        ),
        offload_dir=args.offload_dir,
        statistics_path=args.statistics,
        gpu_memory=args.gpu_memory,
        cpu_memory=args.cpu_memory,
        graph_version=(
            args.graph_version
        ),
    )

    integrated_model = components[
        "integrated_model"
    ]
    backbone = components["backbone"]
    vision_device = components[
        "vision_device"
    ]

    if (
        list(integrated_model.layer_indices)
        != cache_configuration[
            "layer_indices"
        ]
    ):
        raise RuntimeError(
            "Model Attention layers do not "
            "match the First Pass cache."
        )

    if int(
        backbone.num_image_token
    ) != int(
        cache_configuration[
            "tokens_per_tile"
        ]
    ):
        raise RuntimeError(
            "Model image token count does not "
            "match the First Pass cache."
        )

    if cache_configuration.get(
        "adapter_enabled"
    ) is not False:
        raise RuntimeError(
            "Expected an Adapter-OFF "
            "First Pass cache."
        )

    (
        auxiliary_parameters,
        adapter_parameters,
        parameter_counts,
    ) = collect_trainable_parameters(
        integrated_model=integrated_model,
        backbone=backbone,
    )

    trainable_parameters = (
        auxiliary_parameters
        + adapter_parameters
    )

    optimization = (
        create_optimizer_and_scheduler(
            auxiliary_parameters=(
                auxiliary_parameters
            ),
            adapter_parameters=(
                adapter_parameters
            ),
            dataset_size=len(dataset),
            epochs=args.epochs,
            gradient_accumulation_steps=(
                args
                .gradient_accumulation_steps
            ),
            auxiliary_lr=args.auxiliary_lr,
            qlora_lr=args.qlora_lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
        )
    )

    optimizer = optimization["optimizer"]
    scheduler = optimization["scheduler"]
    updates_per_epoch = optimization[
        "updates_per_epoch"
    ]
    total_optimizer_steps = optimization[
        "total_optimizer_steps"
    ]
    warmup_steps = optimization[
        "warmup_steps"
    ]

    print(
        "  Optimizer steps per epoch:",
        updates_per_epoch,
    )
    print(
        "  Total optimizer steps:",
        total_optimizer_steps,
    )
    print(
        "  Warmup steps:",
        warmup_steps,
    )
    print(
        "  Trainable parameters:",
        f"{parameter_counts['total']:,}",
    )

    training_configuration = {
        "runner_schema_version": 1,
        "model": args.model,
        "manifest": str(
            args.manifest.resolve()
        ),
        "manifest_sha256": (
            file_sha256(args.manifest)
        ),
        "statistics": str(
            args.statistics.resolve()
        ),
        "statistics_sha256": (
            file_sha256(args.statistics)
        ),
        "first_pass_cache": str(
            args.first_pass_cache.resolve()
        ),
        "first_pass_configuration": (
            cache_configuration
        ),
        "sample_count": len(dataset),
        "epochs": args.epochs,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": (
            args
            .gradient_accumulation_steps
        ),
        "auxiliary_lr": (
            args.auxiliary_lr
        ),
        "qlora_lr": args.qlora_lr,
        "weight_decay": (
            args.weight_decay
        ),
        "warmup_ratio": (
            args.warmup_ratio
        ),
        "warmup_steps": warmup_steps,
        "max_grad_norm": (
            args.max_grad_norm
        ),
        "seed": args.seed,
        "image_size": image_size,
        "max_num": max_num,
        "supervise_turn_end": True,
        "save_activations_on_cpu": True,
        "qlora": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
            "targets": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],
            "adapter_name": (
                "global_relational"
            ),
        },
        "parameter_counts": (
            parameter_counts
        ),
        "updates_per_epoch": (
            updates_per_epoch
        ),
        "total_optimizer_steps": (
            total_optimizer_steps
        ),
    }

    if args.graph_version == "v1":
        training_configuration[
            "graph_version"
        ] = "v1"

        training_configuration[
            "object_detection_cache"
        ] = {
            "cache_root": str(
                args
                .detection_cache_root
                .expanduser()
                .resolve()
            ),
            "metadata_path": str(
                detection_metadata_path
            ),
            "dataset_manifest": (
                detection_metadata[
                    "dataset_manifest"
                ]
            ),
            "image_id_sha256": (
                detection_metadata[
                    "image_id_sha256"
                ]
            ),
            "unique_images": (
                detection_metadata[
                    "unique_images"
                ]
            ),
            "detector": (
                detection_metadata[
                    "detector"
                ]
            ),
            "configuration": (
                detection_configuration
            ),
        }

    write_json_atomic(
        args.output_dir
        / "run_config.json",
        training_configuration,
    )

    progress = initial_progress()
    metrics = initial_metrics()

    if args.resume is not None:
        print(
            "\nLoading checkpoint:",
            args.resume,
        )

        checkpoint_information = (
            load_training_checkpoint(
                input_path=args.resume,
                integrated_model=(
                    integrated_model
                ),
                optimizer=optimizer,
                scheduler=scheduler,
                expected_configuration=(
                    training_configuration
                ),
                adapter_name=(
                    "global_relational"
                ),
                restore_rng=True,
            )
        )

        progress = dict(
            checkpoint_information[
                "progress"
            ]
        )
        metrics = dict(
            checkpoint_information[
                "metrics"
            ]
        )

        print(
            "  Resume epoch:",
            progress["epoch"],
        )
        print(
            "  Resume batch:",
            progress[
                "next_batch_index"
            ],
        )
        print(
            "  Resume optimizer step:",
            progress["optimizer_step"],
        )

    optimizer.zero_grad(
        set_to_none=True
    )

    if (
        args.max_optimizer_steps > 0
        and progress["optimizer_step"]
        >= args.max_optimizer_steps
    ):
        print(
            "The requested optimizer-step "
            "limit has already been reached."
        )
        return

    resume_epoch = int(
        progress["epoch"]
    )
    training_start = (
        time.perf_counter()
    )
    stop_requested = False

    for epoch in range(
        resume_epoch,
        args.epochs,
    ):
        start_batch_index = (
            int(
                progress[
                    "next_batch_index"
                ]
            )
            if epoch == resume_epoch
            else 0
        )

        if start_batch_index == 0:
            metrics[
                "epoch_loss_sum"
            ] = 0.0
            metrics[
                "epoch_sample_count"
            ] = 0
            metrics[
                "epoch_token_correct"
            ] = 0
            metrics[
                "epoch_token_count"
            ] = 0

        if (
            start_batch_index > 0
            and start_batch_index
            % args.gradient_accumulation_steps
            != 0
        ):
            raise RuntimeError(
                "Resume point is not an "
                "optimizer-step boundary."
            )

        dataloader = build_gqa_dataloader(
            dataset=dataset,
            batch_size=1,
            shuffle=True,
            num_workers=0,
            seed=args.seed,
            epoch=epoch,
        )

        total_batches = len(dataloader)

        print(
            f"\nEpoch {epoch + 1}/"
            f"{args.epochs}, "
            f"starting batch "
            f"{start_batch_index}/"
            f"{total_batches}"
        )

        for batch_index, batch in enumerate(
            dataloader
        ):
            if batch_index < (
                start_batch_index
            ):
                continue

            sample = batch[0]
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
                expected_identity=identity,
                expected_configuration=(
                    cache_configuration
                ),
                tile_size=image_size,
            )

            (
                original_image,
                pixel_values,
                tile_grid,
            ) = load_image(
                image_path=Path(
                    sample["image_path"]
                ),
                image_size=image_size,
                max_num=max_num,
            )

            if tuple(
                first_pass.tile_grid
            ) != tuple(tile_grid):
                raise RuntimeError(
                    "Cached tile grid mismatch "
                    f"for {question_id}."
                )

            if tuple(
                first_pass.image_size
            ) != tuple(
                original_image.size
            ):
                raise RuntimeError(
                    "Cached image size mismatch "
                    f"for {question_id}."
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

                object_cache_path = (
                    detection_cache_path(
                        cache_root=(
                            args
                            .detection_cache_root
                        ),
                        image_id=image_id,
                    )
                )

                detector_metadata = (
                    detection_metadata[
                        "detector"
                    ]
                )

                detection_result = (
                    load_object_detection_cache(
                        input_path=(
                            object_cache_path
                        ),
                        expected_image_id=(
                            image_id
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
                            detection_configuration
                        ),
                    )
                )

                if tuple(
                    detection_result.image_size
                ) != tuple(
                    original_image.size
                ):
                    raise RuntimeError(
                        "Detection image size "
                        "mismatch for "
                        f"{question_id}."
                    )

                detections = (
                    detection_result.detections
                )

            training_output = (
                integrated_model.forward_train(
                    pixel_values=(
                        pixel_values
                    ),
                    question=(
                        sample["question"]
                    ),
                    answer=(
                        sample[
                            "normalized_answer"
                        ]
                    ),
                    tile_grid=tile_grid,
                    image_size=(
                        original_image.size
                    ),
                    supervise_turn_end=True,
                    save_activations_on_cpu=True,
                    first_pass=first_pass,
                    detections=detections,
                )
            )

            loss = training_output.loss

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite loss for "
                    f"{question_id}."
                )

            if not (
                training_output
                .diagnostics
                .get(
                    "first_pass_cached",
                    False,
                )
            ):
                raise RuntimeError(
                    "The supplied First Pass "
                    "cache was not used."
                )

            if (
                training_output
                .diagnostics
                .get("graph_version")
                != args.graph_version
            ):
                raise RuntimeError(
                    "Training Graph version "
                    "diagnostic mismatch."
                )

            if (
                args.graph_version == "v1"
                and training_output
                .diagnostics
                .get("detection_count")
                != len(detections)
            ):
                raise RuntimeError(
                    "Training detection-count "
                    "diagnostic mismatch."
                )

            group_start = (
                batch_index
                // args
                .gradient_accumulation_steps
                * args
                .gradient_accumulation_steps
            )

            group_end = min(
                group_start
                + args
                .gradient_accumulation_steps,
                total_batches,
            )

            group_size = (
                group_end - group_start
            )

            scaled_loss = (
                loss / group_size
            )
            scaled_loss.backward()

            loss_value = float(
                loss.detach()
                .float()
                .item()
            )

            token_correct = int(
                training_output
                .diagnostics[
                    "teacher_forced_token_correct"
                ]
            )
            token_count = int(
                training_output
                .diagnostics[
                    "teacher_forced_token_total"
                ]
            )

            metrics[
                "total_loss_sum"
            ] += loss_value
            metrics[
                "total_sample_count"
            ] += 1
            metrics[
                "total_token_correct"
            ] += token_correct
            metrics[
                "total_token_count"
            ] += token_count

            metrics[
                "epoch_loss_sum"
            ] += loss_value
            metrics[
                "epoch_sample_count"
            ] += 1
            metrics[
                "epoch_token_correct"
            ] += token_correct
            metrics[
                "epoch_token_count"
            ] += token_count

            progress["global_step"] += 1
            progress["samples_seen"] += 1

            update_boundary = (
                batch_index + 1
                == group_end
            )

            if update_boundary:
                unscaled_gradient_norm = (
                    gradient_norm(
                        trainable_parameters
                    )
                )

                if not math.isfinite(
                    unscaled_gradient_norm
                ):
                    raise RuntimeError(
                        "Non-finite gradient "
                        "norm."
                    )

                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_norm=(
                        args.max_grad_norm
                    ),
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(
                    set_to_none=True
                )

                progress[
                    "optimizer_step"
                ] += 1
                progress[
                    "accumulation_remainder"
                ] = 0

                next_batch_index = (
                    batch_index + 1
                )

                if (
                    next_batch_index
                    >= total_batches
                ):
                    progress["epoch"] = (
                        epoch + 1
                    )
                    progress[
                        "next_batch_index"
                    ] = 0
                else:
                    progress["epoch"] = (
                        epoch
                    )
                    progress[
                        "next_batch_index"
                    ] = (
                        next_batch_index
                    )

                epoch_loss = safe_ratio(
                    metrics[
                        "epoch_loss_sum"
                    ],
                    metrics[
                        "epoch_sample_count"
                    ],
                )

                epoch_accuracy = safe_ratio(
                    metrics[
                        "epoch_token_correct"
                    ],
                    metrics[
                        "epoch_token_count"
                    ],
                )

                log_record = {
                    "epoch": epoch,
                    "batch_index": (
                        batch_index
                    ),
                    "global_step": (
                        progress[
                            "global_step"
                        ]
                    ),
                    "optimizer_step": (
                        progress[
                            "optimizer_step"
                        ]
                    ),
                    "question_id": (
                        question_id
                    ),
                    "loss": loss_value,
                    "epoch_loss": (
                        epoch_loss
                    ),
                    "epoch_token_accuracy": (
                        epoch_accuracy
                    ),
                    "gradient_norm": (
                        unscaled_gradient_norm
                    ),
                    "auxiliary_lr": (
                        optimizer
                        .param_groups[0]["lr"]
                    ),
                    "qlora_lr": (
                        optimizer
                        .param_groups[1]["lr"]
                    ),
                }

                append_jsonl(
                    log_path,
                    log_record,
                )

                if (
                    progress[
                        "optimizer_step"
                    ]
                    % args.log_every
                    == 0
                ):
                    elapsed_minutes = (
                        time.perf_counter()
                        - training_start
                    ) / 60

                    print(
                        f"  step="
                        f"{progress['optimizer_step']}/"
                        f"{total_optimizer_steps} "
                        f"batch={batch_index + 1}/"
                        f"{total_batches} "
                        f"loss={loss_value:.6f} "
                        f"avg={epoch_loss:.6f} "
                        f"token_acc="
                        f"{epoch_accuracy:.4f} "
                        f"grad="
                        f"{unscaled_gradient_norm:.4f} "
                        f"time="
                        f"{elapsed_minutes:.1f}m"
                    )

                reached_limit = (
                    args.max_optimizer_steps
                    > 0
                    and progress[
                        "optimizer_step"
                    ]
                    >= args
                    .max_optimizer_steps
                )

                should_checkpoint = (
                    progress[
                        "optimizer_step"
                    ]
                    % args.checkpoint_every
                    == 0
                    or progress[
                        "next_batch_index"
                    ]
                    == 0
                    or reached_limit
                )

                if should_checkpoint:
                    checkpoint_information = (
                        save_runner_checkpoint(
                            output_dir=(
                                args.output_dir
                            ),
                            integrated_model=(
                                integrated_model
                            ),
                            optimizer=optimizer,
                            scheduler=scheduler,
                            configuration=(
                                training_configuration
                            ),
                            progress=progress,
                            metrics=metrics,
                        )
                    )

                    print(
                        "    checkpoint:",
                        checkpoint_information[
                            "path"
                        ],
                    )

                if reached_limit:
                    stop_requested = True

            del scaled_loss
            del loss
            del training_output
            del pixel_values
            del original_image
            del first_pass

            if stop_requested:
                break

        if stop_requested:
            break

        epoch_loss = safe_ratio(
            metrics["epoch_loss_sum"],
            metrics[
                "epoch_sample_count"
            ],
        )
        epoch_accuracy = safe_ratio(
            metrics[
                "epoch_token_correct"
            ],
            metrics[
                "epoch_token_count"
            ],
        )

        print(
            f"Epoch {epoch + 1} complete: "
            f"loss={epoch_loss:.6f}, "
            f"teacher-forced token "
            f"accuracy={epoch_accuracy:.4f}"
        )

    total_loss = safe_ratio(
        metrics["total_loss_sum"],
        metrics["total_sample_count"],
    )
    total_accuracy = safe_ratio(
        metrics["total_token_correct"],
        metrics["total_token_count"],
    )

    peak_cuda_memory = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
        if torch.cuda.is_available()
        else 0.0
    )

    print("\nTraining summary:")
    print(
        "  Status:",
        (
            "STOPPED_AT_RUNTIME_LIMIT"
            if stop_requested
            else "COMPLETE"
        ),
    )
    print(
        "  Optimizer steps:",
        progress["optimizer_step"],
        "/",
        total_optimizer_steps,
    )
    print(
        "  Samples seen:",
        metrics[
            "total_sample_count"
        ],
    )
    print(
        "  Mean training loss:",
        f"{total_loss:.8f}",
    )
    print(
        "  Teacher-forced token accuracy:",
        (
            f"{metrics['total_token_correct']}/"
            f"{metrics['total_token_count']} "
            f"({total_accuracy:.4f})"
        ),
    )
    print(
        "  Peak CUDA memory:",
        f"{peak_cuda_memory:.2f} GiB",
    )
    print(
        "  Output:",
        args.output_dir,
    )

    if stop_requested:
        print(
            "Resume with the checkpoint path "
            "stored in latest_checkpoint.json."
        )
    else:
        print(
            "\nCached QLoRA training run: PASS"
        )
        print(
            "Note: these are optimization "
            "diagnostics, not GQA performance."
        )


if __name__ == "__main__":
    main()