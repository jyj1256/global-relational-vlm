import hashlib
from pathlib import Path
from typing import Mapping, Optional

import torch

from vlm.models.attention_extractor import (
    AttentionExtractionResult,
)
from vlm.models.attention_profile import (
    extract_local_attention_profiles,
)
from vlm.models.attention_reconstructor import (
    reconstruct_local_attention,
)


CACHE_KIND = "global_relational_first_pass"
CACHE_SCHEMA_VERSION = 1


def build_cache_identity(
    question_id,
    image_id,
    question,
):
    if not isinstance(question, str):
        raise TypeError(
            "question must be a string."
        )

    question_hash = hashlib.sha256(
        question.encode("utf-8")
    ).hexdigest()

    return {
        "question_id": str(question_id),
        "image_id": str(image_id),
        "question_sha256": question_hash,
    }


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


def _cpu_tensor(tensor, name):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor."
        )

    return (
        tensor
        .detach()
        .to(device="cpu")
        .contiguous()
    )


def _serialize_attention_result(result):
    return {
        "layer_indices": [
            int(value)
            for value in result.layer_indices
        ],
        "per_layer_head": _cpu_tensor(
            result.per_layer_head,
            "per_layer_head",
        ),
        "image_positions": [
            int(value)
            for value in result.image_positions
        ],
        "local_image_positions": [
            int(value)
            for value
            in result.local_image_positions
        ],
        "thumbnail_positions": [
            int(value)
            for value
            in result.thumbnail_positions
        ],
        "question_start": int(
            result.question_start
        ),
        "question_end": int(
            result.question_end
        ),
        "last_question_token": int(
            result.last_question_token
        ),
        "mean_all_raw": _cpu_tensor(
            result.mean_all_raw,
            "mean_all_raw",
        ),
        "mean_local_raw": _cpu_tensor(
            result.mean_local_raw,
            "mean_local_raw",
        ),
        "mean_thumbnail_raw": _cpu_tensor(
            result.mean_thumbnail_raw,
            "mean_thumbnail_raw",
        ),
        "all_image_l1": _cpu_tensor(
            result.all_image_l1,
            "all_image_l1",
        ),
        "local_image_l1": _cpu_tensor(
            result.local_image_l1,
            "local_image_l1",
        ),
    }


def _deserialize_attention_result(state):
    required_fields = {
        "layer_indices",
        "per_layer_head",
        "image_positions",
        "local_image_positions",
        "thumbnail_positions",
        "question_start",
        "question_end",
        "last_question_token",
        "mean_all_raw",
        "mean_local_raw",
        "mean_thumbnail_raw",
        "all_image_l1",
        "local_image_l1",
    }

    missing = required_fields - set(state)

    if missing:
        raise ValueError(
            "Attention cache fields are missing: "
            + ", ".join(sorted(missing))
        )

    return AttentionExtractionResult(
        layer_indices=[
            int(value)
            for value in state["layer_indices"]
        ],
        per_layer_head=_cpu_tensor(
            state["per_layer_head"],
            "per_layer_head",
        ),
        image_positions=[
            int(value)
            for value
            in state["image_positions"]
        ],
        local_image_positions=[
            int(value)
            for value
            in state["local_image_positions"]
        ],
        thumbnail_positions=[
            int(value)
            for value
            in state["thumbnail_positions"]
        ],
        question_start=int(
            state["question_start"]
        ),
        question_end=int(
            state["question_end"]
        ),
        last_question_token=int(
            state["last_question_token"]
        ),
        mean_all_raw=_cpu_tensor(
            state["mean_all_raw"],
            "mean_all_raw",
        ),
        mean_local_raw=_cpu_tensor(
            state["mean_local_raw"],
            "mean_local_raw",
        ),
        mean_thumbnail_raw=_cpu_tensor(
            state["mean_thumbnail_raw"],
            "mean_thumbnail_raw",
        ),
        all_image_l1=_cpu_tensor(
            state["all_image_l1"],
            "all_image_l1",
        ),
        local_image_l1=_cpu_tensor(
            state["local_image_l1"],
            "local_image_l1",
        ),
    )


def _validate_first_pass_components(
    attention_result,
    attention_profiles,
    projected_visual_features,
    tile_grid,
):
    local_tile_count = (
        tile_grid[0] * tile_grid[1]
    )

    local_token_count = int(
        attention_result
        .local_image_l1
        .numel()
    )

    if local_tile_count <= 0:
        raise ValueError(
            "Local tile count must be positive."
        )

    if local_token_count % local_tile_count != 0:
        raise ValueError(
            "Local Attention token count does not "
            "match the tile grid."
        )

    tokens_per_tile = (
        local_token_count // local_tile_count
    )

    if attention_profiles.ndim != 3:
        raise ValueError(
            "Attention profiles must have shape "
            "(local_tokens, layers, heads)."
        )

    if attention_profiles.shape[0] != (
        local_token_count
    ):
        raise ValueError(
            "Attention profile token count does "
            "not match local Attention."
        )

    if projected_visual_features.ndim != 3:
        raise ValueError(
            "Projected visual features must have "
            "shape (tiles, tokens, hidden)."
        )

    valid_tile_counts = {
        local_tile_count,
        local_tile_count + 1,
    }

    if projected_visual_features.shape[0] not in (
        valid_tile_counts
    ):
        raise ValueError(
            "Projected visual feature tile count "
            "does not match the tile grid."
        )

    if projected_visual_features.shape[1] != (
        tokens_per_tile
    ):
        raise ValueError(
            "Projected visual tokens per tile do "
            "not match local Attention."
        )

    finite_tensors = {
        "per_layer_head": (
            attention_result.per_layer_head
        ),
        "local_image_l1": (
            attention_result.local_image_l1
        ),
        "attention_profiles": (
            attention_profiles
        ),
        "projected_visual_features": (
            projected_visual_features
        ),
    }

    for name, tensor in finite_tensors.items():
        if not torch.isfinite(
            tensor.float()
        ).all():
            raise ValueError(
                f"{name} contains NaN or Inf."
            )


def save_first_pass_cache(
    output_path,
    first_pass,
    identity,
    configuration,
):
    output_path = Path(output_path)

    identity = _canonicalize(identity)
    configuration = _canonicalize(
        configuration
    )

    attention_result = (
        first_pass.attention_result
    )

    derived_profiles = (
        extract_local_attention_profiles(
            attention_result
        )
    )

    original_profiles = _cpu_tensor(
        first_pass.local_attention_profiles,
        "local_attention_profiles",
    )

    if not torch.equal(
        derived_profiles,
        original_profiles,
    ):
        maximum_error = (
            derived_profiles
            - original_profiles
        ).abs().max().item()

        raise ValueError(
            "Stored Attention profiles cannot be "
            "reproduced from per-layer Attention. "
            f"Maximum error: {maximum_error}"
        )

    projected_visual_features = _cpu_tensor(
        first_pass.projected_visual_features,
        "projected_visual_features",
    )

    tile_grid = tuple(
        int(value)
        for value in first_pass.tile_grid
    )
    image_size = tuple(
        int(value)
        for value in first_pass.image_size
    )

    _validate_first_pass_components(
        attention_result=attention_result,
        attention_profiles=derived_profiles,
        projected_visual_features=(
            projected_visual_features
        ),
        tile_grid=tile_grid,
    )

    payload = {
        "cache_kind": CACHE_KIND,
        "schema_version": (
            CACHE_SCHEMA_VERSION
        ),
        "identity": identity,
        "configuration": configuration,
        "first_pass": {
            "attention_result": (
                _serialize_attention_result(
                    attention_result
                )
            ),
            "projected_visual_features": (
                projected_visual_features
            ),
            "tile_grid": list(tile_grid),
            "image_size": list(image_size),
            "profile_shape": list(
                derived_profiles.shape
            ),
        },
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

    return output_path


def load_first_pass_cache(
    input_path,
    expected_identity: Optional[
        Mapping
    ] = None,
    expected_configuration: Optional[
        Mapping
    ] = None,
    tile_size=448,
):
    from vlm.models.global_relational_vlm import (
        FirstPassOutput,
    )

    input_path = Path(input_path)

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    payload = torch.load(
        input_path,
        map_location="cpu",
        weights_only=False,
    )

    if payload.get("cache_kind") != CACHE_KIND:
        raise ValueError(
            "Invalid First Pass cache kind."
        )

    if payload.get("schema_version") != (
        CACHE_SCHEMA_VERSION
    ):
        raise ValueError(
            "Unsupported First Pass cache "
            "schema version."
        )

    if expected_identity is not None:
        expected_identity = _canonicalize(
            expected_identity
        )

        if payload.get("identity") != (
            expected_identity
        ):
            raise ValueError(
                "First Pass cache identity does "
                "not match the requested sample."
            )

    if expected_configuration is not None:
        expected_configuration = _canonicalize(
            expected_configuration
        )

        if payload.get("configuration") != (
            expected_configuration
        ):
            raise ValueError(
                "First Pass cache configuration "
                "does not match the current run."
            )

    state = payload.get("first_pass")

    if not isinstance(state, Mapping):
        raise ValueError(
            "First Pass cache state is missing."
        )

    attention_result = (
        _deserialize_attention_result(
            state["attention_result"]
        )
    )

    attention_profiles = (
        extract_local_attention_profiles(
            attention_result
        )
    )

    projected_visual_features = _cpu_tensor(
        state["projected_visual_features"],
        "projected_visual_features",
    )

    tile_grid = tuple(
        int(value)
        for value in state["tile_grid"]
    )
    image_size = tuple(
        int(value)
        for value in state["image_size"]
    )

    expected_profile_shape = tuple(
        int(value)
        for value in state["profile_shape"]
    )

    if tuple(attention_profiles.shape) != (
        expected_profile_shape
    ):
        raise ValueError(
            "Restored Attention profile shape "
            "does not match the cache."
        )

    _validate_first_pass_components(
        attention_result=attention_result,
        attention_profiles=attention_profiles,
        projected_visual_features=(
            projected_visual_features
        ),
        tile_grid=tile_grid,
    )

    reconstructed_attention = (
        reconstruct_local_attention(
            local_attention=(
                attention_result.local_image_l1
            ),
            tile_grid=tile_grid,
            original_size=image_size,
            tile_size=int(tile_size),
        )
    )

    first_pass = FirstPassOutput(
        attention_result=attention_result,
        local_attention_profiles=(
            attention_profiles
        ),
        reconstructed_attention=(
            reconstructed_attention
        ),
        projected_visual_features=(
            projected_visual_features
        ),
        tile_grid=tile_grid,
        image_size=image_size,
    )

    cache_information = {
        "path": str(input_path),
        "identity": payload["identity"],
        "configuration": (
            payload["configuration"]
        ),
        "schema_version": (
            payload["schema_version"]
        ),
        "file_size_bytes": (
            input_path.stat().st_size
        ),
    }

    return first_pass, cache_information