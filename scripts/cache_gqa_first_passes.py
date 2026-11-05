import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from vlm.data.gqa_dataset import GQADataset
from vlm.models.global_relational_vlm import (
    GlobalRelationalVLM,
)
from vlm.models.qlora_adapter import (
    attach_qlora_adapter,
)
from vlm.scripts.test_backbone import (
    get_vision_device,
    load_image,
)
from vlm.utils.first_pass_cache import (
    build_cache_identity,
    load_first_pass_cache,
    save_first_pass_cache,
)


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "datasets/gqa/subsets/"
    "train_debug_100_seed2026.jsonl"
)
DEFAULT_GQA_ROOT = (
    PROJECT_ROOT
    / "datasets/gqa"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "vlm/outputs/first_pass_cache/"
    "nf4_qlora_maxnum1_train_seed2026"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Cache fixed Adapter-OFF First Passes "
            "for GQA training."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_GQA_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="OpenGVLab/InternVL3-8B",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "pretrained/huggingface"
        ),
    )
    parser.add_argument(
        "--offload-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "pretrained/offload"
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
        type=str,
        default="10GiB",
    )
    parser.add_argument(
        "--cpu-memory",
        type=str,
        default="48GiB",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-new-samples",
        type=int,
        default=0,
        help=(
            "Maximum new caches created in this "
            "run. Zero processes all missing "
            "samples."
        ),
    )

    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        while True:
            block = file.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def cache_path_for_sample(
    entries_dir,
    sample,
):
    return (
        entries_dir
        / f"{sample['question_id']}.pt"
    )


def cache_record(
    output_dir,
    sample,
    first_pass,
    cache_path,
    file_size_bytes,
):
    return {
        "question_id": str(
            sample["question_id"]
        ),
        "image_id": str(
            sample["image_id"]
        ),
        "cache_file": str(
            cache_path.relative_to(
                output_dir
            )
        ),
        "tile_grid": [
            int(value)
            for value in first_pass.tile_grid
        ],
        "image_size": [
            int(value)
            for value in first_pass.image_size
        ],
        "visual_feature_shape": [
            int(value)
            for value in (
                first_pass
                .projected_visual_features
                .shape
            )
        ],
        "file_size_bytes": int(
            file_size_bytes
        ),
    }


def save_metadata(
    metadata_path,
    args,
    configuration,
    manifest_hash,
    dataset_size,
    records,
    new_count,
    reused_count,
    elapsed_seconds,
):
    metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    status = (
        "complete"
        if len(records) == dataset_size
        else "partial"
    )

    total_bytes = sum(
        record["file_size_bytes"]
        for record in records
    )

    metadata = {
        "status": status,
        "dataset": "GQA",
        "split_role": "training",
        "manifest": str(
            args.manifest.resolve()
        ),
        "manifest_sha256": manifest_hash,
        "sample_count": int(
            dataset_size
        ),
        "completed_count": int(
            len(records)
        ),
        "new_count_this_run": int(
            new_count
        ),
        "reused_count_this_run": int(
            reused_count
        ),
        "total_cache_bytes": int(
            total_bytes
        ),
        "elapsed_seconds_this_run": float(
            elapsed_seconds
        ),
        "configuration": configuration,
        "records": records,
    }

    temporary_path = Path(
        str(metadata_path) + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    temporary_path.replace(
        metadata_path
    )


def load_nf4_model(args):
    quantization_config = BitsAndBytesConfig(
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

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir=str(args.cache_dir),
        use_fast=False,
        local_files_only=True,
    )

    print(
        "Loading NF4 InternVL3-8B..."
    )

    backbone = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir=str(args.cache_dir),
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
        offload_folder=str(
            args.offload_dir
        ),
        use_flash_attn=False,
        attn_implementation="eager",
        local_files_only=True,
    ).eval()

    print("Attaching QLoRA Adapter...")

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
        adapter_name="global_relational",
    )

    integrated_model = GlobalRelationalVLM(
        backbone=backbone,
        tokenizer=tokenizer,
    )

    integrated_model.freeze_backbone()
    integrated_model.eval()

    vision_device = get_vision_device(
        backbone
    )

    return (
        integrated_model,
        backbone,
        vision_device,
    )


def main():
    args = parse_args()

    if args.image_size <= 0:
        raise ValueError(
            "--image-size must be positive."
        )

    if args.max_num <= 0:
        raise ValueError(
            "--max-num must be positive."
        )

    if args.progress_every <= 0:
        raise ValueError(
            "--progress-every must be positive."
        )

    if args.max_new_samples < 0:
        raise ValueError(
            "--max-new-samples cannot be negative."
        )

    if not args.manifest.is_file():
        raise FileNotFoundError(
            args.manifest
        )

    dataset = GQADataset(
        manifest_path=args.manifest,
        data_root=args.data_root,
        verify_images=True,
    )

    question_ids = [
        dataset[index]["question_id"]
        for index in range(len(dataset))
    ]

    if len(set(question_ids)) != len(
        question_ids
    ):
        raise ValueError(
            "Manifest contains duplicate "
            "question IDs."
        )

    print("First Pass Cache configuration:")
    print("  Manifest:", args.manifest)
    print("  Samples:", len(dataset))
    print("  Output:", args.output_dir)
    print("  Image size:", args.image_size)
    print("  max_num:", args.max_num)
    print(
        "  Maximum new samples:",
        (
            args.max_new_samples
            if args.max_new_samples > 0
            else "all"
        ),
    )

    (
        integrated_model,
        backbone,
        vision_device,
    ) = load_nf4_model(args)

    configuration = {
        "model": args.model,
        "first_pass_backend": "nf4-qlora",
        "adapter_enabled": False,
        "layer_indices": list(
            integrated_model.layer_indices
        ),
        "image_size": int(
            args.image_size
        ),
        "max_num": int(args.max_num),
        "tokens_per_tile": int(
            backbone.num_image_token
        ),
        "attention_implementation": "eager",
        "local_attention_normalization": (
            "l1"
        ),
    }

    entries_dir = (
        args.output_dir / "entries"
    )
    metadata_dir = (
        args.output_dir / "manifests"
    )
    metadata_path = (
        metadata_dir
        / f"{args.manifest.stem}.json"
    )

    entries_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_hash = file_sha256(
        args.manifest
    )

    records = []
    missing_indices = []
    reused_count = 0

    print(
        "\nScanning existing caches..."
    )

    for index in range(len(dataset)):
        sample = dataset[index]

        identity = build_cache_identity(
            question_id=(
                sample["question_id"]
            ),
            image_id=sample["image_id"],
            question=sample["question"],
        )

        cache_path = cache_path_for_sample(
            entries_dir,
            sample,
        )

        if not cache_path.is_file():
            missing_indices.append(index)
            continue

        try:
            (
                restored_first_pass,
                cache_information,
            ) = load_first_pass_cache(
                input_path=cache_path,
                expected_identity=identity,
                expected_configuration=(
                    configuration
                ),
                tile_size=args.image_size,
            )
        except Exception as error:
            raise RuntimeError(
                "Existing cache validation "
                f"failed: {cache_path}"
            ) from error

        records.append(
            cache_record(
                output_dir=args.output_dir,
                sample=sample,
                first_pass=(
                    restored_first_pass
                ),
                cache_path=cache_path,
                file_size_bytes=(
                    cache_information[
                        "file_size_bytes"
                    ]
                ),
            )
        )

        reused_count += 1

        del restored_first_pass

    print(
        "  Reusable caches:",
        reused_count,
    )
    print(
        "  Missing caches:",
        len(missing_indices),
    )

    start_time = time.perf_counter()
    new_count = 0

    for index in missing_indices:
        if (
            args.max_new_samples > 0
            and new_count
            >= args.max_new_samples
        ):
            break

        sample = dataset[index]

        original_image, pixel_values, tile_grid = (
            load_image(
                image_path=Path(
                    sample["image_path"]
                ),
                image_size=args.image_size,
                max_num=args.max_num,
            )
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
                image_size=(
                    original_image.size
                ),
            )
        )

        identity = build_cache_identity(
            question_id=(
                sample["question_id"]
            ),
            image_id=sample["image_id"],
            question=sample["question"],
        )

        cache_path = cache_path_for_sample(
            entries_dir,
            sample,
        )

        save_first_pass_cache(
            output_path=cache_path,
            first_pass=first_pass,
            identity=identity,
            configuration=configuration,
        )

        (
            restored_first_pass,
            cache_information,
        ) = load_first_pass_cache(
            input_path=cache_path,
            expected_identity=identity,
            expected_configuration=(
                configuration
            ),
            tile_size=args.image_size,
        )

        if tuple(
            restored_first_pass.tile_grid
        ) != tuple(tile_grid):
            raise RuntimeError(
                "Saved cache tile grid does "
                "not match the source."
            )

        if tuple(
            restored_first_pass.image_size
        ) != tuple(original_image.size):
            raise RuntimeError(
                "Saved cache image size does "
                "not match the source."
            )

        records.append(
            cache_record(
                output_dir=args.output_dir,
                sample=sample,
                first_pass=(
                    restored_first_pass
                ),
                cache_path=cache_path,
                file_size_bytes=(
                    cache_information[
                        "file_size_bytes"
                    ]
                ),
            )
        )

        new_count += 1
        completed_count = (
            reused_count + new_count
        )

        if (
            new_count % args.progress_every
            == 0
            or completed_count
            == len(dataset)
        ):
            elapsed = (
                time.perf_counter()
                - start_time
            )

            average_seconds = (
                elapsed / new_count
            )

            remaining = (
                len(dataset)
                - completed_count
            )

            estimated_minutes = (
                average_seconds
                * remaining
                / 60.0
            )

            print(
                f"  [{completed_count:4d}/"
                f"{len(dataset)}] "
                f"question_id="
                f"{sample['question_id']}, "
                f"tile_grid={tile_grid}, "
                f"cache="
                f"{cache_information['file_size_bytes'] / (1024 ** 2):.3f} MiB, "
                f"remaining≈"
                f"{estimated_minutes:.1f} min"
            )

            save_metadata(
                metadata_path=metadata_path,
                args=args,
                configuration=(
                    configuration
                ),
                manifest_hash=(
                    manifest_hash
                ),
                dataset_size=len(dataset),
                records=records,
                new_count=new_count,
                reused_count=(
                    reused_count
                ),
                elapsed_seconds=elapsed,
            )

        del original_image
        del pixel_values
        del first_pass
        del restored_first_pass

        if (
            torch.cuda.is_available()
            and new_count % 20 == 0
        ):
            torch.cuda.empty_cache()

    elapsed = (
        time.perf_counter() - start_time
    )

    records.sort(
        key=lambda record: question_ids.index(
            record["question_id"]
        )
    )

    save_metadata(
        metadata_path=metadata_path,
        args=args,
        configuration=configuration,
        manifest_hash=manifest_hash,
        dataset_size=len(dataset),
        records=records,
        new_count=new_count,
        reused_count=reused_count,
        elapsed_seconds=elapsed,
    )

    completed_count = len(records)
    total_bytes = sum(
        record["file_size_bytes"]
        for record in records
    )

    print("\nFirst Pass Cache summary:")
    print(
        "  Completed:",
        f"{completed_count}/{len(dataset)}",
    )
    print(
        "  Reused:",
        reused_count,
    )
    print(
        "  Newly created:",
        new_count,
    )
    print(
        "  Total size:",
        f"{total_bytes / (1024 ** 2):.3f} MiB",
    )
    print(
        "  Metadata:",
        metadata_path,
    )

    if completed_count == len(dataset):
        print(
            "\nFirst Pass cache generation: "
            "PASS"
        )
    else:
        print(
            "\nFirst Pass cache generation: "
            "PARTIAL"
        )
        print(
            "Run the same command again to "
            "resume."
        )


if __name__ == "__main__":
    main()
