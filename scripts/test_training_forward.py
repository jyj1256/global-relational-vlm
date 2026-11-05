import argparse
import json
import math
import sys
from pathlib import Path

import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from vlm.models.global_relational_vlm import (
    GlobalRelationalVLM,
)
from vlm.scripts.test_backbone import (
    get_vision_device,
    load_image,
)
from vlm.utils.descriptor_statistics import (
    load_descriptor_statistics,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test one real GQA training forward "
            "with a frozen InternVL3 backbone."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default="OpenGVLab/InternVL3-8B",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=(
            "/home/dm-magneto/dm_yunjin/"
            "pretrained/huggingface"
        ),
    )

    parser.add_argument(
        "--offload-dir",
        type=str,
        default=(
            "/home/dm-magneto/dm_yunjin/"
            "vlm/outputs/offload_step23"
        ),
    )

    parser.add_argument(
        "--subset",
        type=str,
        default=(
            "/home/dm-magneto/dm_yunjin/"
            "datasets/gqa/subsets/"
            "val_smoke_20.jsonl"
        ),
    )

    parser.add_argument(
        "--gqa-root",
        type=str,
        default=(
            "/home/dm-magneto/dm_yunjin/"
            "datasets/gqa"
        ),
    )

    parser.add_argument(
        "--statistics",
        type=str,
        default=(
            "/home/dm-magneto/dm_yunjin/"
            "vlm/configs/"
            "gqa_descriptor_statistics_"
            "5000_seed2026.json"
        ),
    )

    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
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
        type=str,
        default="10GiB",
    )

    parser.add_argument(
        "--cpu-memory",
        type=str,
        default="48GiB",
    )

    return parser.parse_args()


def read_jsonl_sample(path, sample_index):
    if sample_index < 0:
        raise ValueError(
            "sample-index must be non-negative."
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        for index, line in enumerate(input_file):
            if index == sample_index:
                return json.loads(line)

    raise IndexError(
        f"Sample index {sample_index} "
        f"was not found in {path}."
    )


def gradient_norm(module):
    squared_norm = 0.0

    for parameter in module.parameters():
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


def has_any_gradient(module):
    return any(
        parameter.grad is not None
        for parameter in module.parameters()
    )


def main():
    args = parse_args()

    torch.manual_seed(2026)

    subset_path = Path(args.subset)
    gqa_root = Path(args.gqa_root)
    statistics_path = Path(args.statistics)
    cache_dir = Path(args.cache_dir)
    offload_dir = Path(args.offload_dir)

    for required_path in (
        subset_path,
        statistics_path,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(
                f"File not found: {required_path}"
            )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    offload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample = read_jsonl_sample(
        path=subset_path,
        sample_index=args.sample_index,
    )

    image_path = (
        gqa_root / sample["image"]
    )

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    question = sample["question"]
    answer = sample["answer"]

    print("GQA training sample:")
    print(
        f"  Question ID: "
        f"{sample['question_id']}"
    )
    print(
        f"  Image ID: {sample['image_id']}"
    )
    print(f"  Image: {image_path}")
    print(f"  Question: {question}")
    print(f"  Answer: {answer}")
    print(
        f"  Type: "
        f"{sample['types']['structural']} / "
        f"{sample['types']['semantic']} / "
        f"{sample['types']['detailed']}"
    )

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir=str(cache_dir),
        use_fast=False,
        local_files_only=True,
    )

    print("Loading InternVL3-8B...")

    backbone = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir=str(cache_dir),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory={
            0: args.gpu_memory,
            "cpu": args.cpu_memory,
        },
        offload_folder=str(offload_dir),
        offload_state_dict=True,
        use_flash_attn=False,
        attn_implementation="eager",
        local_files_only=True,
    ).eval()

    backbone.language_model.config.use_cache = False

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
        backbone
    )

    pixel_values = pixel_values.to(
        device=vision_device,
        dtype=torch.bfloat16,
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
        f"  Vision device: {vision_device}"
    )
    print(
        f"  max_num: {args.max_num} "
        f"(Step-23 smoke setting)"
    )

    integrated_model = GlobalRelationalVLM(
        backbone=backbone,
        tokenizer=tokenizer,
    )

    integrated_model.global_encoder.to(
        device=vision_device,
        dtype=torch.float32,
    )

    integrated_model.graph_encoder.to(
        device=vision_device,
        dtype=torch.float32,
    )

    statistics = load_descriptor_statistics(
        statistics_path
    )

    integrated_model.global_encoder.set_descriptor_statistics(
        mean=statistics["mean"],
        std=statistics["std"],
    )

    integrated_model.freeze_backbone()

    integrated_model.global_encoder.zero_grad(
        set_to_none=True
    )

    integrated_model.graph_encoder.zero_grad(
        set_to_none=True
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(
            vision_device
        )

    print(
        "\nRunning one frozen-backbone "
        "training forward..."
    )

    training_output = (
        integrated_model.forward_train(
            pixel_values=pixel_values,
            question=question,
            answer=answer,
            tile_grid=tile_grid,
            image_size=original_image.size,
            supervise_turn_end=True,
        )
    )

    training_output.global_output.global_token.retain_grad()
    training_output.graph_output.graph_token.retain_grad()

    loss = training_output.loss

    if not torch.isfinite(loss):
        raise RuntimeError(
            "Training loss is not finite."
        )

    print("Running backward...")

    loss.backward()

    global_gradient_norms = {
        "spatial": gradient_norm(
            integrated_model
            .global_encoder
            .spatial_encoder
        ),
        "statistical": gradient_norm(
            integrated_model
            .global_encoder
            .statistical_encoder
        ),
        "fusion": gradient_norm(
            integrated_model
            .global_encoder
            .fusion
        ),
        "projector": gradient_norm(
            integrated_model
            .global_encoder
            .projector
        ),
    }

    graph_gradient_norms = {
        "gnn": gradient_norm(
            integrated_model
            .graph_encoder
            .gnn
        ),
        "graph_head": gradient_norm(
            integrated_model
            .graph_encoder
            .graph_head
        ),
    }

    global_token_gradient = (
        training_output
        .global_output
        .global_token
        .grad
    )

    graph_token_gradient = (
        training_output
        .graph_output
        .graph_token
        .grad
    )

    global_token_gradient_norm = (
        0.0
        if global_token_gradient is None
        else global_token_gradient
        .float()
        .norm()
        .item()
    )

    graph_token_gradient_norm = (
        0.0
        if graph_token_gradient is None
        else graph_token_gradient
        .float()
        .norm()
        .item()
    )

    backbone_has_gradient = (
        has_any_gradient(backbone)
    )

    labels = (
        training_output
        .sequence_batch
        .labels
    )

    supervised_labels = labels[
        labels.ne(-100)
    ]

    turn_end_id = (
        tokenizer.convert_tokens_to_ids(
            "<|im_end|>"
        )
    )

    diagnostics = (
        training_output.diagnostics
    )

    checks = {
        "descriptor_statistics_fitted": (
            integrated_model
            .global_encoder
            .descriptor_statistics_fitted
        ),
        "finite_loss": (
            math.isfinite(loss.item())
        ),
        "positive_loss": (
            loss.item() > 0.0
        ),
        "global_token_shape": (
            tuple(
                training_output
                .global_output
                .global_token
                .shape
            )
            == (1, 1, 3584)
        ),
        "graph_token_shape": (
            tuple(
                training_output
                .graph_output
                .graph_token
                .shape
            )
            == (1, 1, 3584)
        ),
        "input_order": (
            diagnostics["input_order"]
            == (
                "question_global_graph_"
                "image_answer"
            )
        ),
        "answer_supervision": (
            diagnostics[
                "supervised_token_count"
            ]
            == diagnostics[
                "answer_token_count"
            ]
        ),
        "turn_end_supervised": (
            supervised_labels.numel() > 0
            and int(
                supervised_labels[-1].item()
            )
            == int(turn_end_id)
        ),
        "backbone_frozen": (
            diagnostics[
                "backbone_trainable_parameters"
            ]
            == 0
        ),
        "backbone_no_gradient": (
            not backbone_has_gradient
        ),
        "global_token_gradient": (
            global_token_gradient_norm > 0.0
        ),
        "graph_token_gradient": (
            graph_token_gradient_norm > 0.0
        ),
        "spatial_gradient": (
            global_gradient_norms[
                "spatial"
            ] > 0.0
        ),
        "statistical_gradient": (
            global_gradient_norms[
                "statistical"
            ] > 0.0
        ),
        "fusion_gradient": (
            global_gradient_norms[
                "fusion"
            ] > 0.0
        ),
        "projector_gradient": (
            global_gradient_norms[
                "projector"
            ] > 0.0
        ),
        "gnn_gradient": (
            graph_gradient_norms[
                "gnn"
            ] > 0.0
        ),
        "graph_head_gradient": (
            graph_gradient_norms[
                "graph_head"
            ] > 0.0
        ),
    }

    print("\nTraining Forward summary:")
    print(
        f"  Loss: {loss.item():.8f}"
    )
    print(
        f"  Logits shape: "
        f"{diagnostics['logits_shape']}"
    )
    print(
        f"  Original prompt length: "
        f"{diagnostics['original_prompt_length']}"
    )
    print(
        f"  Training sequence length: "
        f"{diagnostics['training_sequence_length']}"
    )
    print(
        f"  Answer tokens: "
        f"{diagnostics['answer_token_count']}"
    )
    print(
        f"  Supervised tokens: "
        f"{diagnostics['supervised_token_count']}"
    )
    print(
        f"  Input order: "
        f"{diagnostics['input_order']}"
    )
    print(
        f"  Global Token index: "
        f"{diagnostics['global_token_index']}"
    )
    print(
        f"  Graph Token index: "
        f"{diagnostics['graph_token_index']}"
    )
    print(
        f"  Image start: "
        f"{diagnostics['reordered_image_start']}"
    )
    print(
        f"  Answer start: "
        f"{diagnostics['answer_start']}"
    )

    print("\nToken gradient norms:")
    print(
        f"  Global Token: "
        f"{global_token_gradient_norm:.8f}"
    )
    print(
        f"  Graph Token: "
        f"{graph_token_gradient_norm:.8f}"
    )

    print("\nGlobal Encoder gradients:")

    for name, norm in (
        global_gradient_norms.items()
    ):
        print(
            f"  {name}: {norm:.8f}"
        )

    print("\nGraph Encoder gradients:")

    for name, norm in (
        graph_gradient_norms.items()
    ):
        print(
            f"  {name}: {norm:.8f}"
        )

    print(
        f"\nBackbone has gradient: "
        f"{backbone_has_gradient}"
    )

    if torch.cuda.is_available():
        peak_memory = (
            torch.cuda
            .max_memory_allocated(
                vision_device
            )
            / (1024 ** 3)
        )

        print(
            f"Peak CUDA memory: "
            f"{peak_memory:.2f} GiB"
        )

    print("\nChecks:")

    for name, passed in checks.items():
        status = (
            "PASS"
            if passed
            else "FAIL"
        )

        print(f"  {name}: {status}")

    if not all(checks.values()):
        failed_checks = [
            name
            for name, passed in checks.items()
            if not passed
        ]

        raise RuntimeError(
            "Training Forward test failed: "
            + ", ".join(failed_checks)
        )

    print(
        "\nGlobal-Relational training "
        "forward test: PASS"
    )
    print(
        "Note: this validates loss and gradient "
        "flow, not task performance."
    )


if __name__ == "__main__":
    main()