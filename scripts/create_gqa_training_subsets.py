import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from vlm.data.gqa_dataset import (
    normalize_gqa_answer,
)
from vlm.scripts.create_gqa_subsets import (
    STRUCTURAL_TYPES,
    interleave,
    load_candidates,
    print_summary,
    select_unique_images,
    write_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic balanced "
            "GQA Train subsets."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/home/dm-magneto/dm_yunjin/"
            "datasets/gqa"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--debug-per-type",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--short-per-type",
        type=int,
        default=200,
    )

    return parser.parse_args()


def load_jsonl_records(path):
    records = []

    if not path.is_file():
        return records

    with path.open(
        "r",
        encoding="utf-8",
    ) as input_file:
        for line in input_file:
            line = line.strip()

            if line:
                records.append(
                    json.loads(line)
                )

    return records


def add_normalized_answers(records):
    changed_count = 0

    for record in records:
        normalized = normalize_gqa_answer(
            record["answer"]
        )

        record["normalized_answer"] = (
            normalized
        )

        if normalized != record["answer"]:
            changed_count += 1

    return changed_count


def question_id_hash(records):
    joined = "\n".join(
        record["question_id"]
        for record in records
    )

    return hashlib.sha256(
        joined.encode("utf-8")
    ).hexdigest()


def count_types(records, type_name):
    return dict(
        Counter(
            record["types"][type_name]
            for record in records
        )
    )


def find_overlap(
    left_records,
    right_records,
    key,
):
    left_values = {
        record[key]
        for record in left_records
    }

    right_values = {
        record[key]
        for record in right_records
    }

    return sorted(
        left_values & right_values
    )


def verify_images(records, root):
    missing = []

    for record in records:
        image_path = (
            root / record["image"]
        )

        if not image_path.is_file():
            missing.append(
                str(image_path)
            )

            if len(missing) >= 10:
                break

    if missing:
        raise FileNotFoundError(
            "Missing selected images:\n"
            + "\n".join(missing)
        )


def main():
    args = parse_args()

    if args.debug_per_type <= 0:
        raise ValueError(
            "debug-per-type must be positive."
        )

    if (
        args.short_per_type
        < args.debug_per_type
    ):
        raise ValueError(
            "short-per-type must be greater "
            "than or equal to debug-per-type."
        )

    root = args.root.resolve()

    train_question_path = (
        root
        / "questions"
        / "train_balanced_questions.json"
    )

    subset_dir = root / "subsets"

    val_debug_path = (
        subset_dir / "val_debug_100.jsonl"
    )

    calibration_path = (
        subset_dir
        / (
            "train_descriptor_calibration_"
            "5000_seed2026.jsonl"
        )
    )

    print(
        f"Loading Train questions: "
        f"{train_question_path}"
    )

    candidates = load_candidates(
        train_question_path
    )

    for structural_type in STRUCTURAL_TYPES:
        print(
            f"  {structural_type}: "
            f"{len(candidates[structural_type]):,} "
            "candidates"
        )

    selected = select_unique_images(
        candidates=candidates,
        per_type=args.short_per_type,
        seed=args.seed,
    )

    debug_records = interleave(
        selected=selected,
        per_type=args.debug_per_type,
    )

    short_records = interleave(
        selected=selected,
        per_type=args.short_per_type,
    )

    debug_changed_answers = (
        add_normalized_answers(
            debug_records
        )
    )

    short_changed_answers = (
        add_normalized_answers(
            short_records
        )
    )

    verify_images(
        records=short_records,
        root=root,
    )

    debug_path = (
        subset_dir
        / (
            "train_debug_100_"
            f"seed{args.seed}.jsonl"
        )
    )

    short_path = (
        subset_dir
        / (
            "train_short_1000_"
            f"seed{args.seed}.jsonl"
        )
    )

    write_jsonl(
        debug_records,
        debug_path,
    )

    write_jsonl(
        short_records,
        short_path,
    )

    val_records = load_jsonl_records(
        val_debug_path
    )

    calibration_records = (
        load_jsonl_records(
            calibration_path
        )
    )

    debug_ids = {
        record["question_id"]
        for record in debug_records
    }

    short_ids = {
        record["question_id"]
        for record in short_records
    }

    if not debug_ids.issubset(
        short_ids
    ):
        raise RuntimeError(
            "Debug subset is not nested "
            "inside the Short-run subset."
        )

    train_val_question_overlap = (
        find_overlap(
            short_records,
            val_records,
            "question_id",
        )
    )

    train_val_image_overlap = (
        find_overlap(
            short_records,
            val_records,
            "image_id",
        )
    )

    calibration_question_overlap = (
        find_overlap(
            short_records,
            calibration_records,
            "question_id",
        )
    )

    if train_val_question_overlap:
        raise RuntimeError(
            "Train and Validation Question IDs "
            "overlap."
        )

    metadata = {
        "source": train_question_path.name,
        "seed": args.seed,
        "selection": (
            "balanced by structural type "
            "with globally unique images"
        ),
        "structural_types": list(
            STRUCTURAL_TYPES
        ),
        "answer_normalization": (
            "NFKC + strip + lowercase + "
            "collapse whitespace"
        ),
        "subsets": {
            debug_path.name: {
                "samples": len(
                    debug_records
                ),
                "question_id_sha256": (
                    question_id_hash(
                        debug_records
                    )
                ),
                "structural_types": (
                    count_types(
                        debug_records,
                        "structural",
                    )
                ),
                "semantic_types": (
                    count_types(
                        debug_records,
                        "semantic",
                    )
                ),
                "normalized_answer_changes": (
                    debug_changed_answers
                ),
            },
            short_path.name: {
                "samples": len(
                    short_records
                ),
                "question_id_sha256": (
                    question_id_hash(
                        short_records
                    )
                ),
                "structural_types": (
                    count_types(
                        short_records,
                        "structural",
                    )
                ),
                "semantic_types": (
                    count_types(
                        short_records,
                        "semantic",
                    )
                ),
                "normalized_answer_changes": (
                    short_changed_answers
                ),
            },
        },
        "debug_nested_in_short": True,
        "train_val_question_overlap": len(
            train_val_question_overlap
        ),
        "train_val_image_overlap": len(
            train_val_image_overlap
        ),
        "calibration_question_overlap": len(
            calibration_question_overlap
        ),
        "validation_manifest": str(
            val_debug_path
        ),
        "calibration_manifest": str(
            calibration_path
        ),
    }

    metadata_path = (
        subset_dir
        / (
            "train_subset_metadata_"
            f"seed{args.seed}.json"
        )
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print_summary(
        "Train Debug subset",
        debug_records,
    )

    print_summary(
        "Train Short-run subset",
        short_records,
    )

    print("\nSplit checks:")
    print(
        "  Debug nested in Short-run:",
        True,
    )
    print(
        "  Train/Val question overlap:",
        len(train_val_question_overlap),
    )
    print(
        "  Train/Val image overlap:",
        len(train_val_image_overlap),
    )
    print(
        "  Calibration question overlap:",
        len(calibration_question_overlap),
    )
    print(
        "  Normalized answer changes "
        "(Debug/Short):",
        (
            debug_changed_answers,
            short_changed_answers,
        ),
    )

    print("\nSaved files:")
    print(f"  {debug_path}")
    print(f"  {short_path}")
    print(f"  {metadata_path}")


if __name__ == "__main__":
    main()