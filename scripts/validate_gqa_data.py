import argparse
from collections import Counter
from pathlib import Path

import ijson


EXPECTED_COUNTS = {
    "train": 943_000,
    "val": 132_062,
    "testdev": 12_578,
}

REQUIRED_FIELDS = (
    "imageId",
    "question",
    "answer",
    "isBalanced",
    "types",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate GQA balanced questions and image connections."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/dm-magneto/dm_yunjin/datasets/gqa"),
    )
    return parser.parse_args()


def validate_split(split, question_path, image_ids):
    total = 0
    missing_fields = Counter()
    missing_images = 0
    missing_image_examples = []
    empty_questions = 0
    empty_answers = 0
    non_balanced = 0
    unique_image_ids = set()
    structural_types = Counter()
    semantic_types = Counter()
    first_example = None

    with question_path.open("rb") as file:
        for question_id, item in ijson.kvitems(file, ""):
            total += 1

            if first_example is None:
                first_example = {
                    "question_id": question_id,
                    "image_id": item.get("imageId"),
                    "question": item.get("question"),
                    "answer": item.get("answer"),
                    "types": item.get("types"),
                }

            for field in REQUIRED_FIELDS:
                if field not in item:
                    missing_fields[field] += 1

            image_id = str(item.get("imageId", ""))
            unique_image_ids.add(image_id)

            if image_id not in image_ids:
                missing_images += 1
                if len(missing_image_examples) < 5:
                    missing_image_examples.append(image_id)

            if not str(item.get("question", "")).strip():
                empty_questions += 1

            if not str(item.get("answer", "")).strip():
                empty_answers += 1

            if item.get("isBalanced") is not True:
                non_balanced += 1

            question_types = item.get("types") or {}
            structural_types[str(question_types.get("structural", "missing"))] += 1
            semantic_types[str(question_types.get("semantic", "missing"))] += 1

            if total % 100_000 == 0:
                print(f"  {split}: checked {total:,} questions")

    expected = EXPECTED_COUNTS[split]
    count_pass = total == expected

    print(f"\n[{split}]")
    print(f"  Question file: {question_path}")
    print(f"  Questions: {total:,}")
    print(f"  Expected: {expected:,}")
    print(f"  Count check: {'PASS' if count_pass else 'FAIL'}")
    print(f"  Unique images used: {len(unique_image_ids):,}")
    print(f"  Missing images: {missing_images:,}")
    print(f"  Missing fields: {dict(missing_fields)}")
    print(f"  Empty questions: {empty_questions:,}")
    print(f"  Empty answers: {empty_answers:,}")
    print(f"  Non-balanced entries: {non_balanced:,}")
    print(f"  Structural types: {dict(structural_types)}")
    print(f"  Semantic types: {dict(semantic_types)}")
    print(f"  First example: {first_example}")

    if missing_image_examples:
        print(f"  Missing image examples: {missing_image_examples}")

    return (
        count_pass
        and missing_images == 0
        and not missing_fields
        and empty_questions == 0
        and empty_answers == 0
        and non_balanced == 0
    )


def main():
    args = parse_args()
    root = args.root.resolve()
    image_dir = root / "images"
    question_dir = root / "questions"

    image_ids = {path.stem for path in image_dir.glob("*.jpg")}

    print(f"GQA root: {root}")
    print(f"Available images: {len(image_ids):,}")

    all_passed = True

    for split in ("train", "val", "testdev"):
        question_path = question_dir / f"{split}_balanced_questions.json"

        if not question_path.is_file():
            print(f"\n[{split}] Missing question file: {question_path}")
            all_passed = False
            continue

        split_passed = validate_split(
            split=split,
            question_path=question_path,
            image_ids=image_ids,
        )
        all_passed = all_passed and split_passed

    print("\nOverall validation:", "PASS" if all_passed else "FAIL")
    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()