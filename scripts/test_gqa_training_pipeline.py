from collections import Counter
from pathlib import Path
import sys


IMPORT_ROOT = Path(__file__).resolve().parents[2]

if str(IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPORT_ROOT))


from vlm.data.gqa_dataset import (
    GQADataset,
    build_gqa_dataloader,
    normalize_gqa_answer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GQA_ROOT = PROJECT_ROOT / "datasets/gqa"
SUBSET_ROOT = GQA_ROOT / "subsets"

STRUCTURAL_TYPES = (
    "query",
    "verify",
    "choose",
    "logical",
    "compare",
)


def structural_counts(dataset):
    return Counter(
        dataset[index]["types"]["structural"]
        for index in range(len(dataset))
    )


def collect_ids(dataset, field):
    return {
        dataset[index][field]
        for index in range(len(dataset))
    }


def normalized_answers_valid(dataset):
    for index in range(len(dataset)):
        sample = dataset[index]

        expected = normalize_gqa_answer(
            sample["answer"]
        )

        if not expected:
            return False

        if sample.get("normalized_answer") != expected:
            return False

    return True


def collect_loader_order(
    dataset,
    seed,
    epoch,
):
    loader = build_gqa_dataloader(
        dataset=dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        seed=seed,
        epoch=epoch,
    )

    order = []

    for batch in loader:
        order.extend(
            sample["question_id"]
            for sample in batch
        )

    return order


def print_dataset_summary(name, dataset):
    summary = dataset.summary()

    print(f"\n{name}:")
    print("  Samples:", summary["samples"])
    print(
        "  Unique images:",
        summary["unique_images"],
    )
    print(
        "  Structural types:",
        summary["structural_types"],
    )
    print(
        "  Semantic types:",
        summary["semantic_types"],
    )


def main():
    debug_dataset = GQADataset(
        manifest_path=(
            SUBSET_ROOT
            / "train_debug_100_seed2026.jsonl"
        ),
        data_root=GQA_ROOT,
        verify_images=True,
    )

    short_dataset = GQADataset(
        manifest_path=(
            SUBSET_ROOT
            / "train_short_1000_seed2026.jsonl"
        ),
        data_root=GQA_ROOT,
        verify_images=True,
    )

    val_dataset = GQADataset(
        manifest_path=(
            SUBSET_ROOT
            / "val_debug_100.jsonl"
        ),
        data_root=GQA_ROOT,
        verify_images=True,
    )

    print_dataset_summary(
        "Train Debug",
        debug_dataset,
    )
    print_dataset_summary(
        "Train Short-run",
        short_dataset,
    )
    print_dataset_summary(
        "Validation Debug",
        val_dataset,
    )

    debug_question_ids = collect_ids(
        debug_dataset,
        "question_id",
    )
    short_question_ids = collect_ids(
        short_dataset,
        "question_id",
    )
    val_question_ids = collect_ids(
        val_dataset,
        "question_id",
    )

    debug_image_ids = collect_ids(
        debug_dataset,
        "image_id",
    )
    short_image_ids = collect_ids(
        short_dataset,
        "image_id",
    )
    val_image_ids = collect_ids(
        val_dataset,
        "image_id",
    )

    debug_expected_types = {
        structural_type: 20
        for structural_type in STRUCTURAL_TYPES
    }
    short_expected_types = {
        structural_type: 200
        for structural_type in STRUCTURAL_TYPES
    }

    epoch_0_order_a = collect_loader_order(
        dataset=short_dataset,
        seed=2026,
        epoch=0,
    )
    epoch_0_order_b = collect_loader_order(
        dataset=short_dataset,
        seed=2026,
        epoch=0,
    )
    epoch_1_order = collect_loader_order(
        dataset=short_dataset,
        seed=2026,
        epoch=1,
    )

    print("\nSplit diagnostics:")
    print(
        "  Debug nested in Short-run:",
        debug_question_ids.issubset(
            short_question_ids
        ),
    )
    print(
        "  Train/Val question overlap:",
        len(
            short_question_ids
            & val_question_ids
        ),
    )
    print(
        "  Train/Val image overlap:",
        len(
            short_image_ids
            & val_image_ids
        ),
    )

    print("\nDataLoader diagnostics:")
    print(
        "  Epoch 0 reproducible:",
        epoch_0_order_a == epoch_0_order_b,
    )
    print(
        "  Epoch changes order:",
        epoch_0_order_a != epoch_1_order,
    )
    print(
        "  Epoch sample sets equal:",
        set(epoch_0_order_a)
        == set(epoch_1_order),
    )
    print(
        "  Epoch 0 samples:",
        len(epoch_0_order_a),
    )
    print(
        "  Epoch 1 samples:",
        len(epoch_1_order),
    )
    print(
        "  Epoch 0 first five:",
        epoch_0_order_a[:5],
    )
    print(
        "  Epoch 1 first five:",
        epoch_1_order[:5],
    )

    checks = {
        "debug_count": (
            len(debug_dataset) == 100
        ),
        "short_count": (
            len(short_dataset) == 1000
        ),
        "val_count": (
            len(val_dataset) == 100
        ),
        "debug_structural_balance": (
            dict(structural_counts(debug_dataset))
            == debug_expected_types
        ),
        "short_structural_balance": (
            dict(structural_counts(short_dataset))
            == short_expected_types
        ),
        "val_structural_balance": (
            dict(structural_counts(val_dataset))
            == debug_expected_types
        ),
        "debug_unique_images": (
            len(debug_image_ids) == 100
        ),
        "short_unique_images": (
            len(short_image_ids) == 1000
        ),
        "val_unique_images": (
            len(val_image_ids) == 100
        ),
        "debug_nested_in_short": (
            debug_question_ids.issubset(
                short_question_ids
            )
        ),
        "train_val_question_disjoint": (
            not (
                short_question_ids
                & val_question_ids
            )
        ),
        "train_val_image_disjoint": (
            not (
                short_image_ids
                & val_image_ids
            )
        ),
        "debug_normalized_answers": (
            normalized_answers_valid(
                debug_dataset
            )
        ),
        "short_normalized_answers": (
            normalized_answers_valid(
                short_dataset
            )
        ),
        "val_normalized_answers": (
            normalized_answers_valid(
                val_dataset
            )
        ),
        "normalization_behavior": (
            normalize_gqa_answer(
                "  Light   Blue  "
            )
            == "light blue"
        ),
        "epoch_0_reproducible": (
            epoch_0_order_a
            == epoch_0_order_b
        ),
        "epoch_changes_order": (
            epoch_0_order_a
            != epoch_1_order
        ),
        "epoch_sample_sets_equal": (
            set(epoch_0_order_a)
            == set(epoch_1_order)
            == short_question_ids
        ),
        "loader_preserves_count": (
            len(epoch_0_order_a)
            == len(short_dataset)
            and len(epoch_1_order)
            == len(short_dataset)
        ),
    }

    print("\nChecks:")

    for name, passed in checks.items():
        print(
            f"  {name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    failed_checks = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    if failed_checks:
        raise RuntimeError(
            "GQA training pipeline checks failed: "
            + ", ".join(failed_checks)
        )

    print(
        "\nGQA Training Data Pipeline test: PASS"
    )
    print(
        "Step 24-5 complete."
    )
    print(
        "Note: this validates data construction and "
        "deterministic loading, not task performance."
    )


if __name__ == "__main__":
    main()