import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from transformers import (
    get_cosine_schedule_with_warmup,
)

from vlm.models.qlora_adapter import (
    get_trainable_adapter_parameters,
)
from vlm.utils.training_checkpoint import (
    save_training_checkpoint,
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def load_json(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def write_json_atomic(path, data):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = Path(
        str(path) + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    temporary_path.replace(path)


def append_jsonl(path, record):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
        )
        file.write("\n")


def load_cache_manifest(
    cache_root,
    manifest_path,
    dataset,
):
    cache_root = Path(cache_root)
    manifest_path = Path(manifest_path)

    metadata_path = (
        cache_root
        / "manifests"
        / f"{manifest_path.stem}.json"
    )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            metadata_path
        )

    metadata = load_json(
        metadata_path
    )

    if metadata.get("status") != "complete":
        raise ValueError(
            "First Pass cache manifest is "
            "not complete."
        )

    if int(
        metadata.get(
            "sample_count",
            -1,
        )
    ) != len(dataset):
        raise ValueError(
            "First Pass cache sample count "
            "does not match the dataset."
        )

    if metadata.get(
        "manifest_sha256"
    ) != file_sha256(manifest_path):
        raise ValueError(
            "First Pass cache manifest hash "
            "does not match the dataset."
        )

    records = metadata.get("records")

    if not isinstance(records, list):
        raise ValueError(
            "First Pass cache records are "
            "missing."
        )

    record_by_question_id = {}

    for record in records:
        question_id = str(
            record["question_id"]
        )

        if question_id in (
            record_by_question_id
        ):
            raise ValueError(
                "Duplicate question ID in "
                "cache metadata: "
                f"{question_id}"
            )

        cache_path = (
            cache_root
            / record["cache_file"]
        )

        if not cache_path.is_file():
            raise FileNotFoundError(
                cache_path
            )

        record_by_question_id[
            question_id
        ] = record

    dataset_question_ids = {
        str(
            dataset[index][
                "question_id"
            ]
        )
        for index in range(len(dataset))
    }

    if set(
        record_by_question_id
    ) != dataset_question_ids:
        raise ValueError(
            "Cache metadata question IDs "
            "do not match the dataset."
        )

    return (
        metadata,
        record_by_question_id,
        metadata_path,
    )


def collect_trainable_parameters(
    integrated_model,
    backbone,
):
    global_parameters = [
        parameter
        for parameter
        in integrated_model
        .global_encoder
        .parameters()
        if parameter.requires_grad
    ]

    graph_parameters = [
        parameter
        for parameter
        in integrated_model
        .graph_encoder
        .parameters()
        if parameter.requires_grad
    ]

    adapter_parameters = (
        get_trainable_adapter_parameters(
            backbone
        )
    )

    auxiliary_parameters = (
        global_parameters
        + graph_parameters
    )

    actual_counts = {
        "global": sum(
            parameter.numel()
            for parameter
            in global_parameters
        ),
        "graph": sum(
            parameter.numel()
            for parameter
            in graph_parameters
        ),
        "adapter": sum(
            parameter.numel()
            for parameter
            in adapter_parameters
        ),
    }

    actual_counts["total"] = sum(
        actual_counts.values()
    )

    expected_counts = {
        "global": 888_469,
        "graph": 1_909_510,
        "adapter": 5_046_272,
        "total": 7_844_251,
    }

    if actual_counts != expected_counts:
        raise RuntimeError(
            "Unexpected trainable parameter "
            f"counts: {actual_counts}"
        )

    return (
        auxiliary_parameters,
        adapter_parameters,
        actual_counts,
    )


def create_optimizer_and_scheduler(
    auxiliary_parameters,
    adapter_parameters,
    dataset_size,
    epochs,
    gradient_accumulation_steps,
    auxiliary_lr,
    qlora_lr,
    weight_decay,
    warmup_ratio,
):
    updates_per_epoch = math.ceil(
        dataset_size
        / gradient_accumulation_steps
    )

    total_optimizer_steps = (
        updates_per_epoch * epochs
    )

    warmup_steps = int(
        round(
            total_optimizer_steps
            * warmup_ratio
        )
    )

    if (
        warmup_ratio > 0
        and total_optimizer_steps > 1
    ):
        warmup_steps = max(
            1,
            warmup_steps,
        )

    warmup_steps = min(
        warmup_steps,
        max(
            0,
            total_optimizer_steps - 1,
        ),
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": (
                    auxiliary_parameters
                ),
                "lr": auxiliary_lr,
                "weight_decay": (
                    weight_decay
                ),
                "group_name": (
                    "auxiliary"
                ),
            },
            {
                "params": (
                    adapter_parameters
                ),
                "lr": qlora_lr,
                "weight_decay": (
                    weight_decay
                ),
                "group_name": "qlora",
            },
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    scheduler = (
        get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=(
                warmup_steps
            ),
            num_training_steps=(
                total_optimizer_steps
            ),
        )
    )

    return {
        "optimizer": optimizer,
        "scheduler": scheduler,
        "updates_per_epoch": (
            updates_per_epoch
        ),
        "total_optimizer_steps": (
            total_optimizer_steps
        ),
        "warmup_steps": warmup_steps,
    }


def gradient_norm(parameters):
    squared_sum = 0.0

    for parameter in parameters:
        if parameter.grad is None:
            continue

        norm = (
            parameter.grad
            .detach()
            .float()
            .norm()
            .item()
        )

        squared_sum += norm ** 2

    return math.sqrt(
        squared_sum
    )


def save_runner_checkpoint(
    output_dir,
    integrated_model,
    optimizer,
    scheduler,
    configuration,
    progress,
    metrics,
):
    output_dir = Path(output_dir)

    checkpoint_path = (
        output_dir
        / "checkpoints"
        / (
            "step_"
            f"{progress['optimizer_step']:06d}"
            ".pt"
        )
    )

    checkpoint_information = (
        save_training_checkpoint(
            output_path=checkpoint_path,
            integrated_model=(
                integrated_model
            ),
            optimizer=optimizer,
            scheduler=scheduler,
            configuration=configuration,
            progress=progress,
            metrics=metrics,
            adapter_name=(
                "global_relational"
            ),
        )
    )

    latest_pointer = {
        "checkpoint": str(
            checkpoint_path.resolve()
        ),
        "optimizer_step": int(
            progress["optimizer_step"]
        ),
        "epoch": int(
            progress["epoch"]
        ),
        "next_batch_index": int(
            progress[
                "next_batch_index"
            ]
        ),
        "file_size_bytes": int(
            checkpoint_information[
                "file_size_bytes"
            ]
        ),
    }

    write_json_atomic(
        output_dir
        / "latest_checkpoint.json",
        latest_pointer,
    )

    return checkpoint_information