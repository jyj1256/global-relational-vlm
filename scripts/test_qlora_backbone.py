import argparse
import math
import sys
from pathlib import Path

import bitsandbytes as bnb
import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from vlm.models.attention_extractor import (
    SelectiveAttentionCollector,
)
from vlm.models.graph_node_builder import (
    extract_projected_visual_features,
)
from vlm.scripts.test_backbone import (
    get_vision_device,
    load_image,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test InternVL3-8B with an NF4 "
            "4-bit language model and BF16 vision."
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
            "vlm/outputs/offload_qlora_test"
        ),
    )
    parser.add_argument(
        "--image",
        default=(
            "/home/dm-magneto/dm_yunjin/"
            "vlm/tests/assets/candy.jpg"
        ),
    )
    parser.add_argument(
        "--question",
        default=(
            "What animal is shown "
            "on the candy?"
        ),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=448,
    )
    parser.add_argument(
        "--max-num",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gpu-memory",
        default="10GiB",
    )
    parser.add_argument(
        "--cpu-memory",
        default="48GiB",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    image_path = Path(args.image)
    cache_dir = Path(args.cache_dir)
    offload_dir = Path(args.offload_dir)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

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
        "Loading InternVL3-8B "
        "with NF4 4-bit LLM..."
    )

    model = AutoModel.from_pretrained(
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

    quantized_module_names = [
        name
        for name, module
        in model.named_modules()
        if isinstance(
            module,
            bnb.nn.Linear4bit,
        )
    ]

    language_quantized_names = [
        name
        for name in quantized_module_names
        if name.startswith(
            "language_model."
        )
    ]

    outside_language_names = [
        name
        for name in quantized_module_names
        if not name.startswith(
            "language_model."
        )
    ]

    vision_bf16_linear_count = sum(
        1
        for name, module
        in model.named_modules()
        if (
            name.startswith("vision_model.")
            and isinstance(module, nn.Linear)
            and not isinstance(
                module,
                bnb.nn.Linear4bit,
            )
        )
    )

    mlp1_bf16_linear_count = sum(
        1
        for name, module
        in model.named_modules()
        if (
            name.startswith("mlp1.")
            and isinstance(module, nn.Linear)
            and not isinstance(
                module,
                bnb.nn.Linear4bit,
            )
        )
    )

    print("\nQuantization scope:")
    print(
        f"  Total Linear4bit modules: "
        f"{len(quantized_module_names)}"
    )
    print(
        f"  Language Linear4bit modules: "
        f"{len(language_quantized_names)}"
    )
    print(
        f"  Outside-language Linear4bit: "
        f"{len(outside_language_names)}"
    )
    print(
        f"  Vision BF16 Linear modules: "
        f"{vision_bf16_linear_count}"
    )
    print(
        f"  mlp1 BF16 Linear modules: "
        f"{mlp1_bf16_linear_count}"
    )
    print(
        f"  is_loaded_in_4bit: "
        f"{getattr(model, 'is_loaded_in_4bit', False)}"
    )

    if outside_language_names:
        print(
            "\nUnexpected quantized modules:"
        )

        for name in outside_language_names[
            :20
        ]:
            print(f"  {name}")

    (
        original_image,
        pixel_values,
        tile_grid,
    ) = load_image(
        image_path=image_path,
        image_size=args.image_size,
        max_num=args.max_num,
    )

    vision_device = get_vision_device(
        model
    )

    pixel_values = pixel_values.to(
        device=vision_device,
        dtype=torch.bfloat16,
    )

    local_tile_count = (
        tile_grid[0] * tile_grid[1]
    )

    total_tile_count = int(
        pixel_values.shape[0]
    )

    tokens_per_tile = int(
        model.num_image_token
    )

    print("\nInput information:")
    print(
        f"  Original size: "
        f"{original_image.size}"
    )
    print(f"  Tile grid: {tile_grid}")
    print(
        f"  Pixel tensor: "
        f"{tuple(pixel_values.shape)}"
    )
    print(
        f"  Vision device: "
        f"{vision_device}"
    )

    print(
        "\nExtracting projected "
        "visual features..."
    )

    with torch.inference_mode():
        visual_features = (
            extract_projected_visual_features(
                model=model,
                pixel_values=pixel_values,
            )
        )

    collector = SelectiveAttentionCollector(
        model=model,
        tokenizer=tokenizer,
        question=args.question,
        layer_indices=tuple(
            range(10, 18)
        ),
        total_tile_count=(
            total_tile_count
        ),
        local_tile_count=(
            local_tile_count
        ),
        tokens_per_tile=(
            tokens_per_tile
        ),
    )

    print(
        "Running quantized First Pass "
        "with eager Attention..."
    )

    collector.install()

    try:
        with torch.inference_mode():
            response = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=(
                    f"<image>\n"
                    f"{args.question}"
                ),
                generation_config={
                    "max_new_tokens": 8,
                    "do_sample": False,
                    "num_beams": 1,
                },
            )
    finally:
        collector.remove()

    attention_result = (
        collector.build_result()
    )

    local_attention = (
        attention_result.local_image_l1
    )

    attention_sum = (
        local_attention.float()
        .sum()
        .item()
    )

    model_footprint_gib = (
        model.get_memory_footprint()
        / (1024 ** 3)
    )

    cuda_allocated_gib = (
        torch.cuda.memory_allocated(
            vision_device
        )
        / (1024 ** 3)
    )

    checks = {
        "model_loaded_in_4bit": bool(
            getattr(
                model,
                "is_loaded_in_4bit",
                False,
            )
        ),
        "language_quantized": (
            len(language_quantized_names) > 0
        ),
        "vision_not_quantized": (
            vision_bf16_linear_count > 0
        ),
        "mlp1_not_quantized": (
            mlp1_bf16_linear_count > 0
        ),
        "nothing_else_quantized": (
            len(outside_language_names) == 0
        ),
        "visual_feature_shape": (
            visual_features.ndim == 3
            and visual_features.shape[0]
            == total_tile_count
            and visual_features.shape[1]
            == tokens_per_tile
            and visual_features.shape[2]
            == 3584
        ),
        "visual_features_finite": bool(
            torch.isfinite(
                visual_features
            ).all()
        ),
        "attention_shape": (
            tuple(
                attention_result
                .per_layer_head.shape
            )
            == (
                8,
                28,
                total_tile_count
                * tokens_per_tile,
            )
        ),
        "local_attention_shape": (
            tuple(local_attention.shape)
            == (
                local_tile_count
                * tokens_per_tile,
            )
        ),
        "attention_sum": (
            abs(attention_sum - 1.0)
            < 1e-5
        ),
        "attention_finite": bool(
            torch.isfinite(
                local_attention
            ).all()
        ),
        "response_nonempty": (
            isinstance(response, str)
            and len(response.strip()) > 0
        ),
    }

    print("\nQLoRA backbone summary:")
    print(
        f"  Visual features: "
        f"{tuple(visual_features.shape)}"
    )
    print(
        f"  Per-layer/head Attention: "
        f"{tuple(attention_result.per_layer_head.shape)}"
    )
    print(
        f"  Local Attention: "
        f"{tuple(local_attention.shape)}"
    )
    print(
        f"  Local Attention sum: "
        f"{attention_sum:.8f}"
    )
    print(
        f"  Model footprint: "
        f"{model_footprint_gib:.2f} GiB"
    )
    print(
        f"  CUDA allocated: "
        f"{cuda_allocated_gib:.2f} GiB"
    )
    print(f"  Response: {response}")

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
            "QLoRA backbone test failed: "
            + ", ".join(failed)
        )

    print(
        "\nInternVL3 QLoRA backbone "
        "compatibility test: PASS"
    )
    print(
        "Note: this validates loading and "
        "Attention extraction, not accuracy."
    )


if __name__ == "__main__":
    main()