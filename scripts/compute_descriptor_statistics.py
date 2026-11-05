import argparse
import hashlib
from contextlib import nullcontext
import json
import random
import sys
from pathlib import Path

import ijson
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from vlm.models.attention_descriptors import (
    AttentionDescriptorExtractor,
    DESCRIPTOR_NAMES,
)
from vlm.models.attention_extractor import (
    SelectiveAttentionCollector,
)
from vlm.models.attention_reconstructor import (
    reconstruct_local_attention,
)
from vlm.models.qlora_adapter import (
    attach_qlora_adapter,
    disable_language_adapter,
    has_qlora_adapter,
)
from vlm.scripts.test_backbone import (
    get_vision_device,
    load_image,
)
from vlm.utils.descriptor_statistics import (
    OnlineDescriptorStatistics,
    load_descriptor_statistics,
    save_descriptor_statistics,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute GQA Train descriptor statistics."
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default="OpenGVLab/InternVL3-8B",
    )

    parser.add_argument(
        "--first-pass",
        type=str,
        choices=(
            "bf16",
            "nf4-qlora",
        ),
        default="bf16",
        help=(
            "Backbone condition used to "
            "compute First-Pass attention."
        ),
    )

    parser.add_argument(
        "--questions",
        type=Path,
        default=Path(
            "/home/dm-magneto/dm_yunjin/"
            "datasets/gqa/questions/"
            "train_balanced_questions.json"
        ),
    )

    parser.add_argument(
        "--images",
        type=Path,
        default=Path(
            "/home/dm-magneto/dm_yunjin/"
            "datasets/gqa/images"
        ),
    )

    parser.add_argument(
        "--subset-file",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--sample-count",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--layers",
        type=str,
        default="10,11,12,13,14,15,16,17",
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            "/home/dm-magneto/dm_yunjin/"
            "pretrained/huggingface"
        ),
    )

    parser.add_argument(
        "--offload-dir",
        type=Path,
        default=Path(
            "/home/dm-magneto/dm_yunjin/"
            "pretrained/offload"
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
        default=4,
    )

    parser.add_argument(
        "--gpu-memory",
        type=str,
        default="10GiB",
    )

    parser.add_argument(
        "--cpu-memory",
        type=str,
        default="64GiB",
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--max-new-samples",
        type=int,
        default=0,
        help=(
            "Process at most this many new samples "
            "and stop after saving progress. "
            "Zero means no limit."
        ),
    )

    parser.add_argument(
        "--restart",
        action="store_true",
        help=(
            "Ignore an existing progress checkpoint "
            "and restart from sample zero."
        ),
    )

    return parser.parse_args()


def parse_layer_indices(layer_string):
    layer_indices = [
        int(value.strip())
        for value in layer_string.split(",")
        if value.strip()
    ]

    if not layer_indices:
        raise ValueError(
            "At least one layer is required."
        )

    return layer_indices


def resolve_output_paths(args):
    subset_root = (
        PROJECT_ROOT
        / "datasets/gqa/subsets"
    )

    config_root = (
        PROJECT_ROOT
        / "vlm/configs"
    )

    if args.subset_file is None:
        args.subset_file = (
            subset_root
            / (
                "train_descriptor_calibration_"
                f"{args.sample_count}_"
                f"seed{args.seed}.jsonl"
            )
        )

    if args.output is None:
        if args.first_pass == "nf4-qlora":
            output_prefix = (
                "gqa_descriptor_statistics_"
                "nf4_qlora_"
            )
        else:
            output_prefix = (
                "gqa_descriptor_statistics_"
            )

        args.output = (
            config_root
            / (
                output_prefix
                + f"{args.sample_count}_"
                + f"seed{args.seed}.json"
            )
        )

    if args.checkpoint is None:
        args.checkpoint = Path(
            str(args.output) + ".progress.pt"
        )


def create_reservoir_subset(
    questions_path,
    sample_count,
    seed,
):
    random_generator = random.Random(seed)
    reservoir = []

    with questions_path.open(
        "rb"
    ) as question_file:
        question_iterator = ijson.kvitems(
            question_file,
            ""
        )

        for index, (
            question_id,
            question_data,
        ) in enumerate(question_iterator):
            image_id = question_data.get(
                "imageId"
            )

            question = question_data.get(
                "question"
            )

            answer = question_data.get(
                "answer"
            )

            if (
                image_id is None
                or question is None
                or answer is None
            ):
                raise ValueError(
                    "A GQA question is missing "
                    "a required field."
                )

            record = {
                "question_id": str(question_id),
                "image_id": str(image_id),
                "question": str(question),
                "answer": str(answer),
                "types": question_data.get(
                    "types",
                    {},
                ),
            }

            if len(reservoir) < sample_count:
                reservoir.append(record)

            else:
                replacement_index = (
                    random_generator.randrange(
                        index + 1
                    )
                )

                if replacement_index < sample_count:
                    reservoir[
                        replacement_index
                    ] = record

    if len(reservoir) != sample_count:
        raise RuntimeError(
            f"Requested {sample_count} questions, "
            f"but sampled {len(reservoir)}."
        )

    reservoir.sort(
        key=lambda item: item["question_id"]
    )

    return reservoir


def save_subset(
    subset_path,
    records,
):
    subset_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with subset_path.open(
        "w",
        encoding="utf-8",
    ) as subset_file:
        for record in records:
            subset_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_subset(
    subset_path,
):
    records = []

    with subset_path.open(
        "r",
        encoding="utf-8",
    ) as subset_file:
        for line in subset_file:
            line = line.strip()

            if line:
                records.append(
                    json.loads(line)
                )

    return records


def load_or_create_subset(args):
    if args.subset_file.exists():
        print(
            "Reusing calibration subset:",
            args.subset_file,
        )

        records = load_subset(
            args.subset_file
        )

        if len(records) != args.sample_count:
            raise ValueError(
                "Existing subset count does not "
                "match --sample-count."
            )

        return records

    print(
        "Sampling GQA Train questions with "
        f"seed {args.seed}..."
    )

    records = create_reservoir_subset(
        questions_path=args.questions,
        sample_count=args.sample_count,
        seed=args.seed,
    )

    save_subset(
        subset_path=args.subset_file,
        records=records,
    )

    print(
        "Saved calibration subset:",
        args.subset_file,
    )

    return records


def load_backbone(args):
    args.cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.offload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir=str(args.cache_dir),
        local_files_only=True,
        use_fast=False,
    )

    if args.first_pass == "nf4-qlora":
        print(
            "Loading InternVL3-8B with "
            "NF4 QLoRA First Pass..."
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

        model = AutoModel.from_pretrained(
            args.model,
            trust_remote_code=True,
            cache_dir=str(args.cache_dir),
            local_files_only=True,
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
            offload_folder=(
                str(args.offload_dir)
            ),
            use_flash_attn=False,
            attn_implementation="eager",
        ).eval()

        adapter_report = (
            attach_qlora_adapter(
                backbone=model,
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

        model.eval()

        print(
            "  QLoRA parameters:",
            f"{adapter_report.trainable_adapter_parameters:,}",
        )

    else:
        print(
            "Loading InternVL3-8B "
            "with BF16 First Pass..."
        )

        model = AutoModel.from_pretrained(
            args.model,
            trust_remote_code=True,
            cache_dir=str(args.cache_dir),
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
            max_memory={
                0: args.gpu_memory,
                "cpu": args.cpu_memory,
            },
            offload_folder=(
                str(args.offload_dir)
            ),
            offload_state_dict=True,
            use_flash_attn=False,
            attn_implementation="eager",
        ).eval()

    model.img_context_token_id = (
        tokenizer.convert_tokens_to_ids(
            "<IMG_CONTEXT>"
        )
    )

    return model, tokenizer


def extract_descriptors(
    model,
    tokenizer,
    descriptor_extractor,
    record,
    images_root,
    image_size,
    max_num,
    layer_indices,
):
    image_path = (
        images_root
        / f"{record['image_id']}.jpg"
    )

    if not image_path.exists():
        raise FileNotFoundError(image_path)

    (
        original_image,
        pixel_values,
        tile_grid,
    ) = load_image(
        image_path=str(image_path),
        image_size=image_size,
        max_num=max_num,
    )

    vision_device = get_vision_device(model)

    pixel_values = pixel_values.to(
        device=vision_device,
        dtype=torch.bfloat16,
    )

    local_tile_count = (
        tile_grid[0] * tile_grid[1]
    )

    collector = SelectiveAttentionCollector(
        model=model,
        tokenizer=tokenizer,
        question=record["question"],
        layer_indices=layer_indices,
        total_tile_count=pixel_values.shape[0],
        local_tile_count=local_tile_count,
        tokens_per_tile=model.num_image_token,
    )

    if has_qlora_adapter(model):
        adapter_context = (
            disable_language_adapter(
                model
            )
        )
    else:
        adapter_context = nullcontext()

    collector.install()

    try:
        with adapter_context:
            with torch.inference_mode():
                model.chat(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,
                    question=(
                        f"<image>\n"
                        f"{record['question']}"
                    ),
                    generation_config={
                        "max_new_tokens": 1,
                        "do_sample": False,
                        "num_beams": 1,
                        "pad_token_id": (
                            tokenizer
                            .eos_token_id
                        ),
                    },
                )

    finally:
        collector.remove()

    attention_result = collector.build_result()

    reconstructed = reconstruct_local_attention(
        local_attention=(
            attention_result
            .local_image_l1
            .detach()
            .float()
            .cpu()
        ),
        tile_grid=tile_grid,
        original_size=original_image.size,
        tile_size=image_size,
    )

    with torch.no_grad():
        descriptor_result = descriptor_extractor(
            reconstructed.original_map
        )

    return (
        descriptor_result.descriptors.cpu(),
        tile_grid,
    )


def question_id_hash(records):
    joined_ids = "\n".join(
        record["question_id"]
        for record in records
    )

    return hashlib.sha256(
        joined_ids.encode("utf-8")
    ).hexdigest()


def build_checkpoint_signature(
    args,
    records,
    layer_indices,
):
    return {
        "sample_count": args.sample_count,
        "seed": args.seed,
        "question_id_sha256": (
            question_id_hash(records)
        ),
        "model": args.model,
        "first_pass_backend": (
            args.first_pass
        ),
        "layer_indices": list(layer_indices),
        "image_size": args.image_size,
        "max_num": args.max_num,
        "map_size": 32,
        "top_p": 0.05,
        "peak_threshold": 0.50,
        "max_peak_count": 16,
    }


def save_progress_checkpoint(
    checkpoint_path,
    accumulator,
    next_index,
    tile_grid_counts,
    signature,
):
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = Path(
        str(checkpoint_path) + ".tmp"
    )

    torch.save(
        {
            "signature": signature,
            "next_index": int(next_index),
            "tile_grid_counts": dict(
                tile_grid_counts
            ),
            "accumulator": (
                accumulator.state_dict()
            ),
        },
        temporary_path,
    )

    temporary_path.replace(
        checkpoint_path
    )


def load_progress_checkpoint(
    checkpoint_path,
    accumulator,
    expected_signature,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint.get("signature") != (
        expected_signature
    ):
        raise ValueError(
            "Progress checkpoint configuration "
            "does not match the current run."
        )

    accumulator.load_state_dict(
        checkpoint["accumulator"]
    )

    next_index = int(
        checkpoint["next_index"]
    )

    if accumulator.count != next_index:
        raise ValueError(
            "Checkpoint count and next index "
            "do not match."
        )

    tile_grid_counts = dict(
        checkpoint.get(
            "tile_grid_counts",
            {},
        )
    )

    return next_index, tile_grid_counts


def main():
    args = parse_args()

    if args.sample_count <= 1:
        raise ValueError(
            "--sample-count must be greater than 1."
        )

    if args.checkpoint_every <= 0:
        raise ValueError(
            "--checkpoint-every must be positive."
        )

    if args.max_new_samples < 0:
        raise ValueError(
            "--max-new-samples cannot be negative."
        )

    resolve_output_paths(args)

    if not args.questions.exists():
        raise FileNotFoundError(args.questions)

    if not args.images.exists():
        raise FileNotFoundError(args.images)

    layer_indices = parse_layer_indices(
        args.layers
    )

    records = load_or_create_subset(args)

    model, tokenizer = load_backbone(args)

    descriptor_extractor = (
        AttentionDescriptorExtractor(
            map_size=32,
            top_p=0.05,
            peak_threshold=0.50,
            max_peak_count=16,
        )
    )

    accumulator = OnlineDescriptorStatistics(
        descriptor_names=DESCRIPTOR_NAMES,
        eps=1e-6,
    )

    signature = build_checkpoint_signature(
        args=args,
        records=records,
        layer_indices=layer_indices,
    )

    start_index = 0
    tile_grid_counts = {}

    if (
        args.checkpoint.exists()
        and not args.restart
    ):
        (
            start_index,
            tile_grid_counts,
        ) = load_progress_checkpoint(
            checkpoint_path=args.checkpoint,
            accumulator=accumulator,
            expected_signature=signature,
        )

        print(
            "\nResuming descriptor statistics:"
        )
        print(
            f"  Completed: {start_index}/"
            f"{len(records)}"
        )
        print(
            f"  Next sample: {start_index + 1}"
        )
        print(
            f"  Checkpoint: {args.checkpoint}"
        )

    elif args.restart:
        print(
            "\nRestart requested. Starting from "
            "sample zero."
        )

        save_progress_checkpoint(
            checkpoint_path=args.checkpoint,
            accumulator=accumulator,
            next_index=0,
            tile_grid_counts={},
            signature=signature,
        )

    if start_index > len(records):
        raise RuntimeError(
            "Checkpoint index exceeds subset length."
        )

    print(
        "\nComputing descriptor statistics "
        f"from {len(records)} Train questions..."
    )

    processed_this_run = 0

    for zero_based_index in range(
        start_index,
        len(records),
    ):
        record = records[zero_based_index]

        descriptors, tile_grid = (
            extract_descriptors(
                model=model,
                tokenizer=tokenizer,
                descriptor_extractor=(
                    descriptor_extractor
                ),
                record=record,
                images_root=args.images,
                image_size=args.image_size,
                max_num=args.max_num,
                layer_indices=layer_indices,
            )
        )

        accumulator.update(descriptors)

        tile_key = (
            f"{tile_grid[0]}x{tile_grid[1]}"
        )

        tile_grid_counts[tile_key] = (
            tile_grid_counts.get(tile_key, 0) + 1
        )

        processed_count = zero_based_index + 1
        processed_this_run += 1

        reached_run_limit = (
            args.max_new_samples > 0
            and processed_this_run
            >= args.max_new_samples
            and processed_count < len(records)
        )

        should_checkpoint = (
            processed_count
            % args.checkpoint_every
            == 0
            or processed_count == len(records)
            or reached_run_limit
        )

        if should_checkpoint:
            save_progress_checkpoint(
                checkpoint_path=args.checkpoint,
                accumulator=accumulator,
                next_index=processed_count,
                tile_grid_counts=(
                    tile_grid_counts
                ),
                signature=signature,
            )

        if (
            processed_count
            % args.progress_every
            == 0
            or processed_count == len(records)
            or reached_run_limit
        ):
            print(
                f"  [{processed_count:>5}/"
                f"{len(records)}] "
                f"question_id="
                f"{record['question_id']}, "
                f"image_id={record['image_id']}, "
                f"tile_grid={tile_grid}"
            )

        if reached_run_limit:
            print(
                "\nIntentional early stop after "
                f"{processed_this_run} new samples."
            )
            print(
                "Resume with the same command "
                "without --restart and "
                "--max-new-samples."
            )
            print(
                "Progress checkpoint:",
                args.checkpoint,
            )
            return

    statistics = accumulator.finalize()

    metadata = {
        "dataset": "GQA",
        "split": "train",
        "statistics_scope": (
            "deterministic calibration subset"
        ),
        "sample_count": args.sample_count,
        "seed": args.seed,
        "subset_file": str(args.subset_file),
        "question_id_sha256": (
            question_id_hash(records)
        ),
        "model": args.model,
        "first_pass_backend": (
            args.first_pass
        ),
        "llm_weight_format": (
            "nf4_4bit"
            if args.first_pass
            == "nf4-qlora"
            else "bfloat16"
        ),
        "vision_weight_dtype": (
            "bfloat16"
        ),
        "qlora_adapter_attached": (
            args.first_pass
            == "nf4-qlora"
        ),
        "qlora_adapter_enabled_during_first_pass": (
            False
            if args.first_pass
            == "nf4-qlora"
            else None
        ),
        "qlora_rank": (
            8
            if args.first_pass
            == "nf4-qlora"
            else None
        ),
        "qlora_alpha": (
            16
            if args.first_pass
            == "nf4-qlora"
            else None
        ),
        "qlora_dropout": (
            0.05
            if args.first_pass
            == "nf4-qlora"
            else None
        ),
        "qlora_target_modules": (
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ]
            if args.first_pass
            == "nf4-qlora"
            else None
        ),
        "bnb_4bit_use_double_quant": (
            True
            if args.first_pass
            == "nf4-qlora"
            else None
        ),
        "bnb_4bit_compute_dtype": (
            "bfloat16"
            if args.first_pass
            == "nf4-qlora"
            else None
        ),
        "layer_indices_zero_based": (
            layer_indices
        ),
        "layers_human_readable": [
            index + 1
            for index in layer_indices
        ],
        "attention_reference": (
            "last non-special question token"
        ),
        "attention_normalization": (
            "local-image L1"
        ),
        "map_size": 32,
        "top_p": 0.05,
        "peak_threshold": 0.50,
        "max_peak_count": 16,
        "image_size": args.image_size,
        "max_num": args.max_num,
        "tile_grid_counts": tile_grid_counts,
    }

    save_descriptor_statistics(
        output_path=args.output,
        statistics=statistics,
        metadata=metadata,
    )

    loaded = load_descriptor_statistics(
        input_path=args.output,
        expected_descriptor_names=(
            DESCRIPTOR_NAMES
        ),
    )

    print("\nDescriptor statistics:")

    for index, name in enumerate(
        DESCRIPTOR_NAMES
    ):
        print(
            f"  {name:<26} "
            f"mean={loaded['mean'][index].item():.8f} "
            f"std={loaded['std'][index].item():.8f} "
            f"min={loaded['minimum'][index].item():.8f} "
            f"max={loaded['maximum'][index].item():.8f}"
        )

    clamped_indices = (
        loaded["raw_std"] < loaded["eps"]
    ).nonzero(as_tuple=False).flatten()

    print("\nSummary:")
    print(
        "  First Pass backend:",
        args.first_pass,
    )
    print("  Count:", loaded["count"])
    print(
        "  Tile grids:",
        tile_grid_counts,
    )
    print(
        "  Clamped std descriptors:",
        clamped_indices.tolist(),
    )
    print(
        "  Subset:",
        args.subset_file,
    )
    print(
        "  Statistics:",
        args.output,
    )

    if loaded["count"] != args.sample_count:
        raise RuntimeError(
            "Saved statistic count mismatch."
        )

    if not torch.isfinite(
        loaded["mean"]
    ).all():
        raise RuntimeError(
            "Mean contains NaN or Inf."
        )

    if not torch.isfinite(
        loaded["std"]
    ).all():
        raise RuntimeError(
            "Std contains NaN or Inf."
        )

    print(
        "\nDescriptor statistics computation: PASS"
    )

    if args.sample_count < 5000:
        print(
            "Note: this is a calibration smoke/debug "
            "result, not the final training statistic."
        )


if __name__ == "__main__":
    main()