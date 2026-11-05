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
            "Overfit eight fixed GQA Train samples "
            "with the auxiliary encoders."
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
            "vlm/outputs/offload_overfit8"
        ),
    )
    parser.add_argument(
        "--subset",
        default=(
            "/home/dm-magneto/dm_yunjin/"
            "datasets/gqa/subsets/"
            "train_descriptor_calibration_"
            "5000_seed2026.jsonl"
        ),
    )
    parser.add_argument(
        "--gqa-root",
        default=(
            "/home/dm-magneto/dm_yunjin/"
            "datasets/gqa"
        ),
    )
    parser.add_argument(
        "--statistics",
        default=(
            "/home/dm-magneto/dm_yunjin/"
            "vlm/configs/"
            "gqa_descriptor_statistics_"
            "5000_seed2026.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "/home/dm-magneto/dm_yunjin/"
            "vlm/outputs/checkpoints/"
            "overfit8_auxiliary_test.pt"
        ),
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
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
        default="8GiB",
    )
    parser.add_argument(
        "--cpu-memory",
        default="48GiB",
    )

    return parser.parse_args()


def read_samples(path, count):
    samples = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        for line in input_file:
            samples.append(json.loads(line))

            if len(samples) == count:
                break

    if len(samples) != count:
        raise RuntimeError(
            f"Requested {count} samples but "
            f"found {len(samples)}."
        )

    return samples


def resolve_image_path(
    sample,
    gqa_root,
):
    image_value = sample.get("image")

    if image_value is None:
        image_value = (
            f"images/{sample['image_id']}.jpg"
        )

    image_path = Path(image_value)

    if not image_path.is_absolute():
        image_path = gqa_root / image_path

    return image_path


def trainable_parameters(model):
    return [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]


def parameter_count(parameters):
    return sum(
        parameter.numel()
        for parameter in parameters
    )


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
            parameter.detach().float().cpu()
            - initial
        )

        squared_norm += (
            difference.norm().item() ** 2
        )

    return squared_norm ** 0.5


@torch.no_grad()
def evaluate(
    integrated_model,
    cached_samples,
):
    integrated_model.global_encoder.eval()
    integrated_model.graph_encoder.eval()
    integrated_model.backbone.eval()

    weighted_loss_sum = 0.0
    correct_sum = 0
    token_sum = 0
    all_finite = True

    for cached in cached_samples:
        output = (
            integrated_model.forward_train(
                pixel_values=(
                    cached["pixel_values"]
                ),
                question=(
                    cached["sample"]["question"]
                ),
                answer=(
                    cached["sample"]["answer"]
                ),
                tile_grid=(
                    cached["tile_grid"]
                ),
                image_size=(
                    cached["image_size"]
                ),
                supervise_turn_end=True,
                save_activations_on_cpu=False,
                first_pass=(
                    cached["first_pass"]
                ),
            )
        )

        loss_value = float(
            output.loss.float().item()
        )

        correct = int(
            output.diagnostics[
                "teacher_forced_token_correct"
            ]
        )

        total = int(
            output.diagnostics[
                "teacher_forced_token_total"
            ]
        )

        all_finite = (
            all_finite
            and math.isfinite(loss_value)
        )

        weighted_loss_sum += (
            loss_value * total
        )

        correct_sum += correct
        token_sum += total

        if not output.diagnostics[
            "first_pass_cached"
        ]:
            raise RuntimeError(
                "Evaluation did not use the "
                "cached First Pass."
            )

        del output

    return {
        "loss": weighted_loss_sum / token_sum,
        "accuracy": correct_sum / token_sum,
        "correct": correct_sum,
        "tokens": token_sum,
        "finite": all_finite,
    }


def main():
    args = parse_args()

    torch.manual_seed(2026)

    subset_path = Path(args.subset)
    gqa_root = Path(args.gqa_root)
    statistics_path = Path(args.statistics)
    cache_dir = Path(args.cache_dir)
    offload_dir = Path(args.offload_dir)
    output_path = Path(args.output)

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
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    samples = read_samples(
        path=subset_path,
        count=args.sample_count,
    )

    print("Overfit-8 configuration:")
    print(
        f"  Samples: {len(samples)}"
    )
    print(
        f"  Epochs: {args.epochs}"
    )
    print(
        f"  Learning rate: "
        f"{args.learning_rate}"
    )
    print(
        f"  max_num: {args.max_num}"
    )
    print(
        f"  GPU model allocation: "
        f"{args.gpu_memory}"
    )

    print("\nSelected Train samples:")

    for index, sample in enumerate(samples):
        print(
            f"  {index + 1}. "
            f"question_id={sample['question_id']}, "
            f"image_id={sample['image_id']}, "
            f"answer={sample['answer']!r}"
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

    vision_device = get_vision_device(
        backbone
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

    auxiliary_parameters = (
        trainable_parameters(
            integrated_model.global_encoder
        )
        + trainable_parameters(
            integrated_model.graph_encoder
        )
    )

    print(
        f"\nTrainable auxiliary parameters: "
        f"{parameter_count(auxiliary_parameters):,}"
    )

    if parameter_count(
        auxiliary_parameters
    ) != 2_797_979:
        raise RuntimeError(
            "Unexpected auxiliary parameter count."
        )

    optimizer = torch.optim.AdamW(
        auxiliary_parameters,
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    print(
        "\nPrecomputing frozen First Passes..."
    )

    cached_samples = []

    for index, sample in enumerate(samples):
        image_path = resolve_image_path(
            sample=sample,
            gqa_root=gqa_root,
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        (
            original_image,
            pixel_values,
            tile_grid,
        ) = load_image(
            image_path=image_path,
            image_size=args.image_size,
            max_num=args.max_num,
        )

        pixel_values = pixel_values.to(
            device=vision_device,
            dtype=torch.bfloat16,
        )

        first_pass = (
            integrated_model.extract_first_pass(
                pixel_values=pixel_values,
                question=sample["question"],
                tile_grid=tile_grid,
                image_size=original_image.size,
            )
        )

        cached_samples.append(
            {
                "sample": sample,
                "pixel_values": pixel_values,
                "tile_grid": tile_grid,
                "image_size": (
                    original_image.size
                ),
                "first_pass": first_pass,
            }
        )

        print(
            f"  [{index + 1}/{len(samples)}] "
            f"question_id="
            f"{sample['question_id']}, "
            f"tile_grid={tile_grid}"
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(
            vision_device
        )

    print("\nEvaluating before training...")

    initial_metrics = evaluate(
        integrated_model=integrated_model,
        cached_samples=cached_samples,
    )

    print(
        f"  Initial loss: "
        f"{initial_metrics['loss']:.8f}"
    )
    print(
        f"  Initial token accuracy: "
        f"{initial_metrics['correct']}/"
        f"{initial_metrics['tokens']} "
        f"({initial_metrics['accuracy']:.4f})"
    )

    initial_parameters = [
        parameter.detach()
        .float()
        .cpu()
        .clone()
        for parameter in auxiliary_parameters
    ]

    epoch_history = []

    for epoch in range(1, args.epochs + 1):
        integrated_model.global_encoder.train()
        integrated_model.graph_encoder.train()
        integrated_model.backbone.eval()

        weighted_loss_sum = 0.0
        correct_sum = 0
        token_sum = 0

        print(
            f"\nEpoch {epoch}/{args.epochs}"
        )

        for index, cached in enumerate(
            cached_samples
        ):
            optimizer.zero_grad(
                set_to_none=True
            )

            output = (
                integrated_model.forward_train(
                    pixel_values=(
                        cached["pixel_values"]
                    ),
                    question=(
                        cached["sample"][
                            "question"
                        ]
                    ),
                    answer=(
                        cached["sample"][
                            "answer"
                        ]
                    ),
                    tile_grid=(
                        cached["tile_grid"]
                    ),
                    image_size=(
                        cached["image_size"]
                    ),
                    supervise_turn_end=True,
                    save_activations_on_cpu=True,
                    first_pass=(
                        cached["first_pass"]
                    ),
                )
            )

            loss = output.loss

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite training loss."
                )

            loss.backward()

            gradient_norm = (
                torch.nn.utils
                .clip_grad_norm_(
                    auxiliary_parameters,
                    max_norm=(
                        args.max_grad_norm
                    ),
                )
            )

            if not torch.isfinite(
                gradient_norm
            ):
                raise RuntimeError(
                    "Non-finite gradient norm."
                )

            optimizer.step()

            token_total = int(
                output.diagnostics[
                    "teacher_forced_token_total"
                ]
            )

            token_correct = int(
                output.diagnostics[
                    "teacher_forced_token_correct"
                ]
            )

            loss_value = float(
                loss.detach().float().item()
            )

            weighted_loss_sum += (
                loss_value * token_total
            )

            correct_sum += token_correct
            token_sum += token_total

            print(
                f"  [{index + 1}/"
                f"{len(cached_samples)}] "
                f"loss={loss_value:.6f}, "
                f"accuracy="
                f"{token_correct}/"
                f"{token_total}, "
                f"grad_norm="
                f"{float(gradient_norm):.6f}"
            )

            del output
            del loss

        epoch_loss = (
            weighted_loss_sum / token_sum
        )

        epoch_accuracy = (
            correct_sum / token_sum
        )

        epoch_history.append(
            {
                "epoch": epoch,
                "training_loss": epoch_loss,
                "training_accuracy": (
                    epoch_accuracy
                ),
            }
        )

        print(
            f"  Epoch training loss: "
            f"{epoch_loss:.8f}"
        )
        print(
            f"  Epoch training accuracy: "
            f"{correct_sum}/{token_sum} "
            f"({epoch_accuracy:.4f})"
        )

    print("\nEvaluating after training...")

    final_metrics = evaluate(
        integrated_model=integrated_model,
        cached_samples=cached_samples,
    )

    delta_norm = parameter_delta_norm(
        parameters=auxiliary_parameters,
        initial_parameters=initial_parameters,
    )

    loss_reduction = (
        initial_metrics["loss"]
        - final_metrics["loss"]
    )

    loss_reduction_ratio = (
        loss_reduction
        / initial_metrics["loss"]
    )

    backbone_has_gradient = any(
        parameter.grad is not None
        for parameter
        in integrated_model.backbone.parameters()
    )

    checks = {
        "initial_finite": (
            initial_metrics["finite"]
        ),
        "final_finite": (
            final_metrics["finite"]
        ),
        "loss_decreased": (
            final_metrics["loss"]
            < initial_metrics["loss"]
        ),
        "parameters_updated": (
            delta_norm > 0.0
        ),
        "accuracy_not_worse": (
            final_metrics["accuracy"]
            >= initial_metrics["accuracy"]
        ),
        "backbone_no_gradient": (
            not backbone_has_gradient
        ),
        "statistics_fitted": (
            integrated_model
            .global_encoder
            .descriptor_statistics_fitted
        ),
    }

    artifact = {
        "trained": False,
        "purpose": (
            "Step-23 overfit/debug test"
        ),
        "sample_count": len(samples),
        "epochs": args.epochs,
        "learning_rate": (
            args.learning_rate
        ),
        "question_ids": [
            sample["question_id"]
            for sample in samples
        ],
        "initial_metrics": initial_metrics,
        "epoch_history": epoch_history,
        "final_metrics": final_metrics,
        "loss_reduction": loss_reduction,
        "loss_reduction_ratio": (
            loss_reduction_ratio
        ),
        "parameter_delta_norm": delta_norm,
        "global_encoder_state_dict": (
            integrated_model
            .global_encoder
            .state_dict()
        ),
        "graph_encoder_state_dict": (
            integrated_model
            .graph_encoder
            .state_dict()
        ),
    }

    torch.save(
        artifact,
        output_path,
    )

    print("\nOverfit-8 summary:")
    print(
        f"  Initial loss: "
        f"{initial_metrics['loss']:.8f}"
    )
    print(
        f"  Final loss: "
        f"{final_metrics['loss']:.8f}"
    )
    print(
        f"  Loss reduction: "
        f"{loss_reduction:.8f} "
        f"({loss_reduction_ratio:.2%})"
    )
    print(
        f"  Initial accuracy: "
        f"{initial_metrics['correct']}/"
        f"{initial_metrics['tokens']} "
        f"({initial_metrics['accuracy']:.4f})"
    )
    print(
        f"  Final accuracy: "
        f"{final_metrics['correct']}/"
        f"{final_metrics['tokens']} "
        f"({final_metrics['accuracy']:.4f})"
    )
    print(
        f"  Parameter delta norm: "
        f"{delta_norm:.8f}"
    )
    print(
        f"  Backbone has gradient: "
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
            f"  Peak CUDA memory: "
            f"{peak_memory:.2f} GiB"
        )

    print(f"  Saved artifact: {output_path}")

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
            "Overfit-8 test failed: "
            + ", ".join(failed)
        )

    print("\nOverfit-8 test: PASS")
    print(
        "Note: this is a training-pipeline "
        "debug test, not a performance result."
    )


if __name__ == "__main__":
    main()