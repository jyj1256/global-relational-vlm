import hashlib
import json
import random
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import torch
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)


CHECKPOINT_KIND = (
    "global_relational_qlora_training"
)
CHECKPOINT_SCHEMA_VERSION = 1


def _canonicalize(value):
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: str(pair[0]),
            )
        }

    if isinstance(value, (list, tuple)):
        return [
            _canonicalize(item)
            for item in value
        ]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, torch.dtype):
        return str(value)

    return value


def _configuration_hash(configuration):
    serialized = json.dumps(
        configuration,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


def _cpu_clone(value):
    if isinstance(value, torch.Tensor):
        return (
            value
            .detach()
            .to(device="cpu")
            .clone()
        )

    if isinstance(value, Mapping):
        return {
            key: _cpu_clone(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _cpu_clone(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            _cpu_clone(item)
            for item in value
        )

    return value


def _module_state_on_cpu(module):
    return {
        name: (
            tensor
            .detach()
            .to(device="cpu")
            .clone()
        )
        for name, tensor
        in module.state_dict().items()
    }


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": (
            torch.get_rng_state()
        ),
        "torch_cuda": None,
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = (
            torch.cuda.get_rng_state_all()
        )

    return _cpu_clone(state)


def restore_rng_state(state):
    random.setstate(
        state["python"]
    )
    np.random.set_state(
        state["numpy"]
    )
    torch.set_rng_state(
        state["torch_cpu"]
    )

    cuda_state = state.get(
        "torch_cuda"
    )

    if (
        cuda_state is not None
        and torch.cuda.is_available()
    ):
        torch.cuda.set_rng_state_all(
            cuda_state
        )


def _move_optimizer_state_to_devices(
    optimizer,
):
    for parameter, state in (
        optimizer.state.items()
    ):
        for name, value in state.items():
            if isinstance(
                value,
                torch.Tensor,
            ):
                state[name] = value.to(
                    device=parameter.device
                )


def save_training_checkpoint(
    output_path,
    integrated_model,
    optimizer,
    scheduler,
    configuration,
    progress,
    metrics,
    adapter_name="global_relational",
):
    output_path = Path(output_path)

    configuration = _canonicalize(
        configuration
    )
    progress = _canonicalize(progress)
    metrics = _canonicalize(metrics)

    accumulation_remainder = int(
        progress.get(
            "accumulation_remainder",
            0,
        )
    )

    if accumulation_remainder != 0:
        raise ValueError(
            "Checkpoints must be saved at an "
            "optimizer-step boundary."
        )

    language_model = (
        integrated_model
        .backbone
        .language_model
    )

    adapter_state = (
        get_peft_model_state_dict(
            language_model,
            adapter_name=adapter_name,
            save_embedding_layers=False,
        )
    )

    if not adapter_state:
        raise RuntimeError(
            "QLoRA Adapter state is empty."
        )

    payload = {
        "checkpoint_kind": (
            CHECKPOINT_KIND
        ),
        "schema_version": (
            CHECKPOINT_SCHEMA_VERSION
        ),
        "adapter_name": str(
            adapter_name
        ),
        "configuration": configuration,
        "configuration_sha256": (
            _configuration_hash(
                configuration
            )
        ),
        "progress": progress,
        "metrics": metrics,
        "model_state": {
            "global_encoder": (
                _module_state_on_cpu(
                    integrated_model
                    .global_encoder
                )
            ),
            "graph_encoder": (
                _module_state_on_cpu(
                    integrated_model
                    .graph_encoder
                )
            ),
            "qlora_adapter": (
                _cpu_clone(
                    adapter_state
                )
            ),
        },
        "optimizer_state": (
            _cpu_clone(
                optimizer.state_dict()
            )
        ),
        "scheduler_state": (
            _cpu_clone(
                scheduler.state_dict()
            )
            if scheduler is not None
            else None
        ),
        "rng_state": capture_rng_state(),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = Path(
        str(output_path) + ".tmp"
    )

    torch.save(
        payload,
        temporary_path,
    )

    temporary_path.replace(
        output_path
    )

    return {
        "path": str(output_path),
        "file_size_bytes": (
            output_path.stat().st_size
        ),
        "configuration_sha256": (
            payload[
                "configuration_sha256"
            ]
        ),
    }


def load_training_checkpoint(
    input_path,
    integrated_model,
    optimizer=None,
    scheduler=None,
    expected_configuration: Optional[
        Mapping
    ] = None,
    adapter_name="global_relational",
    restore_rng=True,
):
    input_path = Path(input_path)

    if not input_path.is_file():
        raise FileNotFoundError(
            input_path
        )

    payload = torch.load(
        input_path,
        map_location="cpu",
        weights_only=False,
    )

    if payload.get(
        "checkpoint_kind"
    ) != CHECKPOINT_KIND:
        raise ValueError(
            "Invalid training checkpoint kind."
        )

    if payload.get(
        "schema_version"
    ) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported training checkpoint "
            "schema version."
        )

    stored_configuration = (
        payload["configuration"]
    )

    stored_hash = payload.get(
        "configuration_sha256"
    )

    if stored_hash != _configuration_hash(
        stored_configuration
    ):
        raise ValueError(
            "Checkpoint configuration hash "
            "does not match its contents."
        )

    if expected_configuration is not None:
        expected_configuration = (
            _canonicalize(
                expected_configuration
            )
        )

        if stored_configuration != (
            expected_configuration
        ):
            raise ValueError(
                "Training checkpoint "
                "configuration does not match "
                "the current run."
            )

    stored_adapter_name = payload.get(
        "adapter_name"
    )

    if stored_adapter_name != str(
        adapter_name
    ):
        raise ValueError(
            "Checkpoint Adapter name does "
            "not match the current run."
        )

    model_state = payload.get(
        "model_state"
    )

    if not isinstance(
        model_state,
        Mapping,
    ):
        raise ValueError(
            "Checkpoint model state is missing."
        )

    integrated_model.global_encoder.load_state_dict(
        model_state["global_encoder"],
        strict=True,
    )

    integrated_model.graph_encoder.load_state_dict(
        model_state["graph_encoder"],
        strict=True,
    )

    set_peft_model_state_dict(
        integrated_model
        .backbone
        .language_model,
        model_state["qlora_adapter"],
        adapter_name=adapter_name,
        ignore_mismatched_sizes=False,
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            payload["optimizer_state"]
        )

        _move_optimizer_state_to_devices(
            optimizer
        )

    stored_scheduler_state = (
        payload.get("scheduler_state")
    )

    if (
        scheduler is not None
        and stored_scheduler_state
        is not None
    ):
        scheduler.load_state_dict(
            stored_scheduler_state
        )

    if restore_rng:
        restore_rng_state(
            payload["rng_state"]
        )

    return {
        "path": str(input_path),
        "file_size_bytes": (
            input_path.stat().st_size
        ),
        "configuration": (
            stored_configuration
        ),
        "configuration_sha256": (
            stored_hash
        ),
        "progress": payload["progress"],
        "metrics": payload["metrics"],
        "adapter_name": (
            stored_adapter_name
        ),
    }