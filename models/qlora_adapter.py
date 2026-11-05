from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence, Tuple

import torch.nn as nn
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)


DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
)


@dataclass(frozen=True)
class QLoRAAdapterReport:
    adapter_name: str
    target_modules: Tuple[str, ...]
    rank: int
    alpha: int
    dropout: float
    trainable_adapter_parameters: int
    non_adapter_trainable_parameters: int
    non_language_trainable_parameters: int
    trainable_adapter_tensor_count: int


def _parameter_count(parameters):
    return sum(
        parameter.numel()
        for parameter in parameters
    )


def _is_lora_parameter(name):
    return "lora_" in name


def has_qlora_adapter(backbone):
    return isinstance(
        backbone.language_model,
        PeftModel,
    )


def get_trainable_adapter_parameters(
    backbone,
):
    if not has_qlora_adapter(backbone):
        return []

    return [
        parameter
        for name, parameter
        in backbone.language_model
        .named_parameters()
        if (
            parameter.requires_grad
            and _is_lora_parameter(name)
        )
    ]


def get_trainable_adapter_names(
    backbone,
):
    if not has_qlora_adapter(backbone):
        return []

    return [
        name
        for name, parameter
        in backbone.language_model
        .named_parameters()
        if (
            parameter.requires_grad
            and _is_lora_parameter(name)
        )
    ]


def attach_qlora_adapter(
    backbone,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: Sequence[str] = (
        DEFAULT_TARGET_MODULES
    ),
    adapter_name: str = "global_relational",
):
    """
    Attach one LoRA adapter to the quantized
    InternVL language model.

    The Vision Encoder, mlp1 projector, and
    all base LLM parameters remain frozen.
    """

    if has_qlora_adapter(backbone):
        raise RuntimeError(
            "A PEFT adapter is already attached "
            "to the language model."
        )

    if not bool(
        getattr(
            backbone,
            "is_loaded_in_4bit",
            False,
        )
    ):
        raise RuntimeError(
            "The InternVL backbone must be "
            "loaded in 4-bit before attaching "
            "the QLoRA adapter."
        )

    rank = int(rank)
    alpha = int(alpha)
    dropout = float(dropout)

    if rank <= 0:
        raise ValueError(
            "rank must be positive."
        )

    if alpha <= 0:
        raise ValueError(
            "alpha must be positive."
        )

    if not 0.0 <= dropout < 1.0:
        raise ValueError(
            "dropout must be in [0, 1)."
        )

    target_modules = tuple(
        str(name)
        for name in target_modules
    )

    if not target_modules:
        raise ValueError(
            "target_modules must not be empty."
        )

    prepared_language_model = (
        prepare_model_for_kbit_training(
            backbone.language_model,
            use_gradient_checkpointing=False,
        )
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(
            target_modules
        ),
        bias="none",
    )

    peft_language_model = get_peft_model(
        prepared_language_model,
        lora_config,
        adapter_name=adapter_name,
    )

    backbone.language_model = (
        peft_language_model
    )

    for name, parameter in (
        backbone.named_parameters()
    ):
        if not name.startswith(
            "language_model."
        ):
            parameter.requires_grad_(False)

    trainable_adapter_parameters = [
        parameter
        for name, parameter
        in backbone.language_model
        .named_parameters()
        if (
            parameter.requires_grad
            and _is_lora_parameter(name)
        )
    ]

    non_adapter_trainable_parameters = [
        parameter
        for name, parameter
        in backbone.language_model
        .named_parameters()
        if (
            parameter.requires_grad
            and not _is_lora_parameter(name)
        )
    ]

    non_language_trainable_parameters = [
        parameter
        for name, parameter
        in backbone.named_parameters()
        if (
            parameter.requires_grad
            and not name.startswith(
                "language_model."
            )
        )
    ]

    report = QLoRAAdapterReport(
        adapter_name=adapter_name,
        target_modules=target_modules,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        trainable_adapter_parameters=(
            _parameter_count(
                trainable_adapter_parameters
            )
        ),
        non_adapter_trainable_parameters=(
            _parameter_count(
                non_adapter_trainable_parameters
            )
        ),
        non_language_trainable_parameters=(
            _parameter_count(
                non_language_trainable_parameters
            )
        ),
        trainable_adapter_tensor_count=len(
            trainable_adapter_parameters
        ),
    )

    if (
        report
        .trainable_adapter_parameters
        <= 0
    ):
        raise RuntimeError(
            "No trainable LoRA parameters "
            "were created."
        )

    if (
        report
        .non_adapter_trainable_parameters
        != 0
    ):
        raise RuntimeError(
            "Unexpected trainable base LLM "
            "parameters were found."
        )

    if (
        report
        .non_language_trainable_parameters
        != 0
    ):
        raise RuntimeError(
            "Vision or multimodal projector "
            "parameters remain trainable."
        )

    return report


def enable_language_adapter(
    backbone,
    adapter_name: str = (
        "global_relational"
    ),
):
    if not has_qlora_adapter(backbone):
        raise RuntimeError(
            "No PEFT adapter is attached."
        )

    backbone.language_model.set_adapter(
        adapter_name
    )


@contextmanager
def disable_language_adapter(backbone):
    """
    Temporarily execute the language model
    without the LoRA contribution.

    Use this context for the frozen First Pass.
    """

    if not has_qlora_adapter(backbone):
        raise RuntimeError(
            "No PEFT adapter is attached."
        )

    with (
        backbone.language_model
        .disable_adapter()
    ):
        yield


def count_all_trainable_parameters(
    module: nn.Module,
):
    return _parameter_count(
        parameter
        for parameter in module.parameters()
        if parameter.requires_grad
    )