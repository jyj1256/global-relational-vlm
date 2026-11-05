from pathlib import Path

import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from vlm.models.global_relational_vlm import (
    GlobalRelationalVLM,
)
from vlm.models.qlora_adapter import (
    attach_qlora_adapter,
)
from vlm.scripts.test_backbone import (
    get_vision_device,
)
from vlm.utils.descriptor_statistics import (
    load_descriptor_statistics,
)


def load_training_model(
    model_name,
    model_cache_dir,
    offload_dir,
    statistics_path,
    gpu_memory="10GiB",
    cpu_memory="48GiB",
):
    model_cache_dir = Path(
        model_cache_dir
    )
    offload_dir = Path(
        offload_dir
    )
    statistics_path = Path(
        statistics_path
    )

    if not statistics_path.is_file():
        raise FileNotFoundError(
            statistics_path
        )

    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=(
                torch.bfloat16
            ),
            llm_int8_skip_modules=[
                "vision_model",
                "mlp1",
            ],
        )
    )

    print("Loading tokenizer...")

    tokenizer = (
        AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=str(
                model_cache_dir
            ),
            use_fast=False,
            local_files_only=True,
        )
    )

    print(
        "Loading NF4 InternVL3-8B..."
    )

    backbone = (
        AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=str(
                model_cache_dir
            ),
            torch_dtype=torch.bfloat16,
            quantization_config=(
                quantization_config
            ),
            low_cpu_mem_usage=True,
            device_map="auto",
            max_memory={
                0: gpu_memory,
                "cpu": cpu_memory,
            },
            offload_folder=str(
                offload_dir
            ),
            use_flash_attn=False,
            attn_implementation="eager",
            local_files_only=True,
        )
        .eval()
    )

    print("Attaching QLoRA Adapter...")

    adapter_report = (
        attach_qlora_adapter(
            backbone=backbone,
            rank=8,
            alpha=16,
            dropout=0.05,
            target_modules=(
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ),
            adapter_name=(
                "global_relational"
            ),
        )
    )

    integrated_model = (
        GlobalRelationalVLM(
            backbone=backbone,
            tokenizer=tokenizer,
        )
    )

    vision_device = (
        get_vision_device(
            backbone
        )
    )

    integrated_model.global_encoder.to(
        device=vision_device,
        dtype=torch.float32,
    )
    integrated_model.graph_encoder.to(
        device=vision_device,
        dtype=torch.float32,
    )

    statistics = (
        load_descriptor_statistics(
            statistics_path
        )
    )

    statistics_backend = (
        statistics["metadata"].get(
            "first_pass_backend"
        )
    )

    if statistics_backend != "nf4-qlora":
        raise RuntimeError(
            "Expected NF4 QLoRA descriptor "
            "statistics, received: "
            f"{statistics_backend!r}"
        )

    integrated_model.global_encoder.set_descriptor_statistics(
        mean=statistics["mean"],
        std=statistics["std"],
    )

    integrated_model.freeze_backbone()

    integrated_model.global_encoder.train()
    integrated_model.graph_encoder.train()

    if not (
        integrated_model
        .global_encoder
        .descriptor_statistics_fitted
    ):
        raise RuntimeError(
            "Descriptor statistics were not "
            "applied to the Global Encoder."
        )

    return {
        "integrated_model": (
            integrated_model
        ),
        "backbone": backbone,
        "tokenizer": tokenizer,
        "vision_device": vision_device,
        "adapter_report": (
            adapter_report
        ),
        "statistics_metadata": (
            statistics["metadata"]
        ),
    }