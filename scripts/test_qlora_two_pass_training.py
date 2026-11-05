import json
import math
import sys
from pathlib import Path

import bitsandbytes as bnb
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from vlm.models.global_relational_vlm import (
    GlobalRelationalVLM,
)
from vlm.models.qlora_adapter import (
    attach_qlora_adapter,
    get_trainable_adapter_parameters,
)
from vlm.scripts.test_backbone import (
    get_vision_device,
    load_image,
)
from vlm.utils.descriptor_statistics import (
    load_descriptor_statistics,
)


MODEL_NAME = "OpenGVLab/InternVL3-8B"

CACHE_DIR = Path(
    "/home/dm-magneto/dm_yunjin/"
    "pretrained/huggingface"
)

OFFLOAD_DIR = Path(
    "/home/dm-magneto/dm_yunjin/"
    "vlm/outputs/offload_qlora_two_pass"
)

SUBSET_PATH = Path(
    "/home/dm-magneto/dm_yunjin/"
    "datasets/gqa/subsets/"
    "val_smoke_20.jsonl"
)

GQA_ROOT = Path(
    "/home/dm-magneto/dm_yunjin/"
    "datasets/gqa"
)

STATISTICS_PATH = Path(
    "/home/dm-magneto/dm_yunjin/"
    "vlm/configs/"
    "gqa_descriptor_statistics_"
    "nf4_qlora_5000_seed2026.json"
)


def read_first_sample(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        return json.loads(
            input_file.readline()
        )


def gradient_norm(parameters):
    squared_norm = 0.0

    for parameter in parameters:
        if parameter.grad is None:
            continue

        value = (
            parameter.grad.detach()
            .float()
            .norm()
            .item()
        )

        squared_norm += value ** 2

    return squared_norm ** 0.5


def module_gradient_norm(module):
    return gradient_norm(
        list(module.parameters())
    )


def clone_parameters(parameters):
    return [
        parameter.detach()
        .float()
        .cpu()
        .clone()
        for parameter in parameters
    ]


def parameter_delta_norm(
    parameters,
    initial_parameters,
):
    squared_norm = 0.0

    for parameter, initial in zip(
        parameters,
        initial_parameters,
    ):
        difference = (
            parameter.detach()
            .float()
            .cpu()
            - initial
        )

        squared_norm += (
            difference.norm().item() ** 2
        )

    return squared_norm ** 0.5


def main():
    torch.manual_seed(2026)

    for required_path in (
        SUBSET_PATH,
        STATISTICS_PATH,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(
                f"File not found: {required_path}"
            )

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    OFFLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample = read_first_sample(
        SUBSET_PATH
    )

    image_path = (
        GQA_ROOT / sample["image"]
    )

    print("QLoRA Two-Pass sample:")
    print(
        f"  Question ID: "
        f"{sample['question_id']}"
    )
    print(
        f"  Image ID: "
        f"{sample['image_id']}"
    )
    print(
        f"  Question: "
        f"{sample['question']}"
    )
    print(
        f"  Answer: {sample['answer']}"
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

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        cache_dir=str(CACHE_DIR),
        use_fast=False,
        local_files_only=True,
    )

    print(
        "Loading NF4 InternVL3-8B..."
    )

    backbone = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        cache_dir=str(CACHE_DIR),
        torch_dtype=torch.bfloat16,
        quantization_config=(
            quantization_config
        ),
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory={
            0: "10GiB",
            "cpu": "48GiB",
        },
        offload_folder=str(
            OFFLOAD_DIR
        ),
        use_flash_attn=False,
        attn_implementation="eager",
        local_files_only=True,
    ).eval()

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

    vision_device = get_vision_device(
        backbone
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
            STATISTICS_PATH
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

    adapter_parameters = (
        get_trainable_adapter_parameters(
            backbone
        )
    )

    auxiliary_parameters = [
        parameter
        for module in (
            integrated_model.global_encoder,
            integrated_model.graph_encoder,
        )
        for parameter in module.parameters()
        if parameter.requires_grad
    ]

    all_trainable_parameters = (
        auxiliary_parameters
        + adapter_parameters
    )

    expected_auxiliary_parameters = (
        2_797_979
    )

    expected_adapter_parameters = (
        5_046_272
    )

    expected_total_parameters = (
        7_844_251
    )

    auxiliary_parameter_count = sum(
        parameter.numel()
        for parameter
        in auxiliary_parameters
    )

    adapter_parameter_count = sum(
        parameter.numel()
        for parameter
        in adapter_parameters
    )

    total_parameter_count = sum(
        parameter.numel()
        for parameter
        in all_trainable_parameters
    )

    print("\nTrainable parameters:")
    print(
        f"  Global + Graph: "
        f"{auxiliary_parameter_count:,}"
    )
    print(
        f"  QLoRA: "
        f"{adapter_parameter_count:,}"
    )
    print(
        f"  Total: "
        f"{total_parameter_count:,}"
    )

    (
        original_image,
        pixel_values,
        tile_grid,
    ) = load_image(
        image_path=image_path,
        image_size=448,
        max_num=1,
    )

    pixel_values = pixel_values.to(
        device=vision_device,
        dtype=torch.bfloat16,
    )

    integrated_model.global_encoder.train()
    integrated_model.graph_encoder.train()

    for parameter in all_trainable_parameters:
        parameter.grad = None

    initial_adapter_parameters = (
        clone_parameters(
            adapter_parameters
        )
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": auxiliary_parameters,
                "lr": 1e-4,
            },
            {
                "params": adapter_parameters,
                "lr": 5e-5,
            },
        ],
        weight_decay=0.0,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(
            vision_device
        )

    print(
        "\nRunning QLoRA Two-Pass "
        "training forward..."
    )

    output = (
        integrated_model.forward_train(
            pixel_values=pixel_values,
            question=sample["question"],
            answer=sample["answer"],
            tile_grid=tile_grid,
            image_size=original_image.size,
            supervise_turn_end=True,
            save_activations_on_cpu=True,
            first_pass=None,
        )
    )

    attention_before = (
        output.first_pass
        .attention_result
        .local_image_l1
        .detach()
        .float()
        .cpu()
        .clone()
    )

    output.global_output.global_token.retain_grad()
    output.graph_output.graph_token.retain_grad()

    loss = output.loss

    if not torch.isfinite(loss):
        raise RuntimeError(
            "Training loss is not finite."
        )

    print("Running backward...")

    loss.backward()

    global_gradient_norm = (
        module_gradient_norm(
            integrated_model.global_encoder
        )
    )

    graph_gradient_norm = (
        module_gradient_norm(
            integrated_model.graph_encoder
        )
    )

    adapter_gradient_norm = gradient_norm(
        adapter_parameters
    )

    global_token_gradient_norm = (
        output.global_output
        .global_token.grad
        .float()
        .norm()
        .item()
    )

    graph_token_gradient_norm = (
        output.graph_output
        .graph_token.grad
        .float()
        .norm()
        .item()
    )

    base_llm_has_gradient = any(
        parameter.grad is not None
        for name, parameter
        in backbone.language_model
        .named_parameters()
        if "lora_" not in name
    )

    non_language_has_gradient = any(
        parameter.grad is not None
        for name, parameter
        in backbone.named_parameters()
        if not name.startswith(
            "language_model."
        )
    )

    total_gradient_norm = (
        torch.nn.utils.clip_grad_norm_(
            all_trainable_parameters,
            max_norm=1.0,
        )
    )

    optimizer.step()

    adapter_delta_norm = (
        parameter_delta_norm(
            parameters=adapter_parameters,
            initial_parameters=(
                initial_adapter_parameters
            ),
        )
    )

    diagnostics = output.diagnostics

    del output
    del loss

    print(
        "\nRecomputing First Pass after "
        "the QLoRA update..."
    )

    first_pass_after = (
        integrated_model.extract_first_pass(
            pixel_values=pixel_values,
            question=sample["question"],
            tile_grid=tile_grid,
            image_size=original_image.size,
        )
    )

    attention_after = (
        first_pass_after
        .attention_result
        .local_image_l1
        .detach()
        .float()
        .cpu()
    )

    first_pass_attention_error = (
        attention_after
        - attention_before
    ).abs().max().item()

    peak_memory_gib = (
        torch.cuda.max_memory_allocated(
            vision_device
        )
        / (1024 ** 3)
    )

    checks = {
        "auxiliary_parameter_count": (
            auxiliary_parameter_count
            == expected_auxiliary_parameters
        ),
        "adapter_parameter_count": (
            adapter_parameter_count
            == expected_adapter_parameters
        ),
        "total_parameter_count": (
            total_parameter_count
            == expected_total_parameters
        ),
        "finite_loss": math.isfinite(
            diagnostics["loss"]
        ),
        "qlora_attached": (
            diagnostics["qlora_attached"]
        ),
        "qlora_trainable_count": (
            diagnostics[
                "qlora_trainable_parameters"
            ]
            == expected_adapter_parameters
        ),
        "no_unexpected_trainable": (
            diagnostics[
                "unexpected_backbone_trainable"
            ]
            == 0
        ),
        "adapter_training": (
            diagnostics[
                "adapter_training"
            ]
        ),
        "global_gradient": (
            global_gradient_norm > 0.0
        ),
        "graph_gradient": (
            graph_gradient_norm > 0.0
        ),
        "adapter_gradient": (
            adapter_gradient_norm > 0.0
        ),
        "global_token_gradient": (
            global_token_gradient_norm > 0.0
        ),
        "graph_token_gradient": (
            graph_token_gradient_norm > 0.0
        ),
        "adapter_updated": (
            adapter_delta_norm > 0.0
        ),
        "base_llm_no_gradient": (
            not base_llm_has_gradient
        ),
        "vision_and_mlp1_no_gradient": (
            not non_language_has_gradient
        ),
        "first_pass_isolated": (
            first_pass_attention_error
            < 1e-7
        ),
    }

    print("\nQLoRA Two-Pass summary:")
    print(
        f"  Loss: "
        f"{diagnostics['loss']:.8f}"
    )
    print(
        f"  Teacher-forced accuracy: "
        f"{diagnostics['teacher_forced_token_correct']}/"
        f"{diagnostics['teacher_forced_token_total']}"
    )
    print(
        f"  Global gradient norm: "
        f"{global_gradient_norm:.8f}"
    )
    print(
        f"  Graph gradient norm: "
        f"{graph_gradient_norm:.8f}"
    )
    print(
        f"  QLoRA gradient norm: "
        f"{adapter_gradient_norm:.8f}"
    )
    print(
        f"  Global Token gradient: "
        f"{global_token_gradient_norm:.8f}"
    )
    print(
        f"  Graph Token gradient: "
        f"{graph_token_gradient_norm:.8f}"
    )
    print(
        f"  Clipped total gradient: "
        f"{float(total_gradient_norm):.8f}"
    )
    print(
        f"  QLoRA parameter delta: "
        f"{adapter_delta_norm:.8f}"
    )
    print(
        f"  First Pass Attention error: "
        f"{first_pass_attention_error:.10f}"
    )
    print(
        f"  Base LLM has gradient: "
        f"{base_llm_has_gradient}"
    )
    print(
        f"  Vision/mlp1 has gradient: "
        f"{non_language_has_gradient}"
    )
    print(
        f"  Peak CUDA memory: "
        f"{peak_memory_gib:.2f} GiB"
    )
    print(
        "  Statistics source: "
        "NF4 QLoRA final statistics"
    )

    print("\nChecks:")

    for name, passed in checks.items():
        print(
            f"  {name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    if not all(checks.values()):
        failed = [
            name
            for name, passed in checks.items()
            if not passed
        ]

        raise RuntimeError(
            "QLoRA Two-Pass training "
            "test failed: "
            + ", ".join(failed)
        )

    print(
        "\nQLoRA Two-Pass training "
        "test: PASS"
    )
    print(
        "Note: this validates Pass isolation "
        "and gradient flow, not performance."
    )


if __name__ == "__main__":
    main()