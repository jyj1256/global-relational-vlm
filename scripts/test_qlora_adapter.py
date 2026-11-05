import argparse
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


from vlm.models.qlora_adapter import (
    attach_qlora_adapter,
    disable_language_adapter,
    enable_language_adapter,
    get_trainable_adapter_names,
    get_trainable_adapter_parameters,
    has_qlora_adapter,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Attach and test a QLoRA adapter "
            "on InternVL3-8B."
        )
    )

    parser.add_argument(
        "--model",
        default="OpenGVLab/InternVL3-8B",
    )
    parser.add_argument(
        "--cache-dir",
        default=(
            "/home/dm-magneto/dm_yunjin/"
            "pretrained/huggingface"
        ),
    )
    parser.add_argument(
        "--offload-dir",
        default=(
            "/home/dm-magneto/dm_yunjin/"
            "vlm/outputs/offload_qlora_adapter"
        ),
    )
    parser.add_argument(
        "--gpu-memory",
        default="10GiB",
    )
    parser.add_argument(
        "--cpu-memory",
        default="48GiB",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    return parser.parse_args()


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


def gradient_norm(parameters):
    squared_norm = 0.0

    for parameter in parameters:
        if parameter.grad is None:
            continue

        norm = (
            parameter.grad.detach()
            .float()
            .norm()
            .item()
        )

        squared_norm += norm ** 2

    return squared_norm ** 0.5


def run_logits(
    language_model,
    input_ids,
    attention_mask,
):
    output = language_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )

    logits = (
        output.logits.detach()
        .float()
        .cpu()
    )

    del output

    return logits


def main():
    args = parse_args()

    torch.manual_seed(2026)

    cache_dir = Path(args.cache_dir)
    offload_dir = Path(args.offload_dir)

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    offload_dir.mkdir(
        parents=True,
        exist_ok=True,
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

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir=str(cache_dir),
        use_fast=False,
        local_files_only=True,
    )

    print(
        "Loading NF4 InternVL3-8B..."
    )

    backbone = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir=str(cache_dir),
        torch_dtype=torch.bfloat16,
        quantization_config=(
            quantization_config
        ),
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory={
            0: args.gpu_memory,
            "cpu": args.cpu_memory,
        },
        offload_folder=str(offload_dir),
        use_flash_attn=False,
        attn_implementation="eager",
        local_files_only=True,
    ).eval()

    print("Attaching QLoRA adapter...")

    report = attach_qlora_adapter(
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
        adapter_name="global_relational",
    )

    adapter_parameters = (
        get_trainable_adapter_parameters(
            backbone
        )
    )

    adapter_names = (
        get_trainable_adapter_names(
            backbone
        )
    )

    expected_adapter_parameters = (
        5_046_272
    )

    expected_adapter_tensors = 224

    print("\nQLoRA Adapter summary:")
    print(
        f"  Adapter name: "
        f"{report.adapter_name}"
    )
    print(
        f"  Target modules: "
        f"{report.target_modules}"
    )
    print(f"  Rank: {report.rank}")
    print(f"  Alpha: {report.alpha}")
    print(
        f"  Dropout: {report.dropout}"
    )
    print(
        f"  Trainable Adapter parameters: "
        f"{report.trainable_adapter_parameters:,}"
    )
    print(
        f"  Trainable Adapter tensors: "
        f"{report.trainable_adapter_tensor_count}"
    )
    print(
        f"  Trainable base LLM parameters: "
        f"{report.non_adapter_trainable_parameters:,}"
    )
    print(
        f"  Trainable non-language parameters: "
        f"{report.non_language_trainable_parameters:,}"
    )

    print("\nFirst ten trainable names:")

    for name in adapter_names[:10]:
        print(f"  {name}")

    prompt_ids = tokenizer(
        (
            "Question: What color is a "
            "clear daytime sky?\n"
            "Answer:"
        ),
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]

    answer_ids = tokenizer(
        " blue",
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]

    turn_end_id = (
        tokenizer.convert_tokens_to_ids(
            "<|im_end|>"
        )
    )

    answer_ids = torch.cat(
        (
            answer_ids,
            torch.tensor(
                [[int(turn_end_id)]],
                dtype=torch.long,
            ),
        ),
        dim=1,
    )

    input_ids = torch.cat(
        (
            prompt_ids,
            answer_ids,
        ),
        dim=1,
    )

    labels = torch.full_like(
        input_ids,
        fill_value=-100,
    )

    labels[
        :,
        -answer_ids.shape[1]:,
    ] = answer_ids

    embedding_device = next(
        backbone.language_model
        .get_input_embeddings()
        .parameters()
    ).device

    input_ids = input_ids.to(
        embedding_device
    )
    labels = labels.to(
        embedding_device
    )

    attention_mask = torch.ones_like(
        input_ids
    )

    backbone.language_model.eval()

    enable_language_adapter(
        backbone,
        adapter_name="global_relational",
    )

    with torch.no_grad():
        enabled_before = run_logits(
            language_model=(
                backbone.language_model
            ),
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        with disable_language_adapter(
            backbone
        ):
            disabled_before = run_logits(
                language_model=(
                    backbone.language_model
                ),
                input_ids=input_ids,
                attention_mask=(
                    attention_mask
                ),
            )

    initial_identity_error = (
        enabled_before
        - disabled_before
    ).abs().max().item()

    initial_parameters = clone_parameters(
        adapter_parameters
    )

    optimizer = torch.optim.AdamW(
        adapter_parameters,
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    backbone.language_model.train()
    optimizer.zero_grad(
        set_to_none=True
    )

    training_output = (
        backbone.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
    )

    loss = training_output.loss

    if loss is None:
        raise RuntimeError(
            "Language model returned no loss."
        )

    if not torch.isfinite(loss):
        raise RuntimeError(
            "QLoRA test loss is not finite."
        )

    loss.backward()

    adapter_gradient_norm = gradient_norm(
        adapter_parameters
    )

    base_parameter_has_gradient = any(
        parameter.grad is not None
        for name, parameter
        in backbone.language_model
        .named_parameters()
        if not (
            "lora_" in name
        )
    )

    non_language_has_gradient = any(
        parameter.grad is not None
        for name, parameter
        in backbone.named_parameters()
        if not name.startswith(
            "language_model."
        )
    )

    optimizer.step()

    adapter_delta_norm = (
        parameter_delta_norm(
            parameters=adapter_parameters,
            initial_parameters=(
                initial_parameters
            ),
        )
    )

    del training_output

    backbone.language_model.eval()

    with torch.no_grad():
        enable_language_adapter(
            backbone,
            adapter_name=(
                "global_relational"
            ),
        )

        enabled_after = run_logits(
            language_model=(
                backbone.language_model
            ),
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        with disable_language_adapter(
            backbone
        ):
            disabled_after = run_logits(
                language_model=(
                    backbone.language_model
                ),
                input_ids=input_ids,
                attention_mask=(
                    attention_mask
                ),
            )

    base_preservation_error = (
        disabled_after
        - disabled_before
    ).abs().max().item()

    adapter_effect = (
        enabled_after
        - disabled_after
    ).abs().max().item()

    cuda_allocated_gib = (
        torch.cuda.memory_allocated()
        / (1024 ** 3)
    )

    checks = {
        "adapter_attached": (
            has_qlora_adapter(backbone)
        ),
        "adapter_parameter_count": (
            report
            .trainable_adapter_parameters
            == expected_adapter_parameters
        ),
        "adapter_tensor_count": (
            report
            .trainable_adapter_tensor_count
            == expected_adapter_tensors
        ),
        "base_llm_frozen": (
            report
            .non_adapter_trainable_parameters
            == 0
        ),
        "vision_and_mlp1_frozen": (
            report
            .non_language_trainable_parameters
            == 0
        ),
        "initial_adapter_identity": (
            initial_identity_error < 1e-5
        ),
        "finite_loss": bool(
            torch.isfinite(loss)
        ),
        "adapter_gradient": (
            adapter_gradient_norm > 0.0
        ),
        "adapter_updated": (
            adapter_delta_norm > 0.0
        ),
        "base_no_gradient": (
            not base_parameter_has_gradient
        ),
        "non_language_no_gradient": (
            not non_language_has_gradient
        ),
        "base_output_preserved": (
            base_preservation_error < 1e-5
        ),
        "adapter_effect_after_update": (
            adapter_effect > 1e-6
        ),
    }

    print("\nTraining diagnostics:")
    print(
        f"  Loss: {loss.item():.8f}"
    )
    print(
        f"  Adapter gradient norm: "
        f"{adapter_gradient_norm:.8f}"
    )
    print(
        f"  Adapter parameter delta: "
        f"{adapter_delta_norm:.8f}"
    )
    print(
        f"  Base parameter gradient: "
        f"{base_parameter_has_gradient}"
    )
    print(
        f"  Non-language gradient: "
        f"{non_language_has_gradient}"
    )

    print("\nAdapter switching:")
    print(
        f"  Initial identity error: "
        f"{initial_identity_error:.10f}"
    )
    print(
        f"  Disabled base preservation error: "
        f"{base_preservation_error:.10f}"
    )
    print(
        f"  Enabled Adapter effect: "
        f"{adapter_effect:.10f}"
    )
    print(
        f"  CUDA allocated: "
        f"{cuda_allocated_gib:.2f} GiB"
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
            "QLoRA Adapter test failed: "
            + ", ".join(failed)
        )

    print("\nQLoRA Adapter test: PASS")
    print(
        "Note: this validates Adapter "
        "attachment and switching, not "
        "task performance."
    )


if __name__ == "__main__":
    main()