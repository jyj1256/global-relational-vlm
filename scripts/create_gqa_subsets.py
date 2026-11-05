import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import ijson


STRUCTURAL_TYPES = (
    "query",
    "verify",
    "choose",
    "logical",
    "compare",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create deterministic GQA validation subsets."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/dm-magneto/dm_yunjin/datasets/gqa"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    return parser.parse_args()


def stable_key(seed, question_id):
    value = f"{seed}:{question_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def load_candidates(question_path):
    candidates = {
        structural_type: []
        for structural_type in STRUCTURAL_TYPES
    }

    with question_path.open("rb") as file:
        for question_id, item in ijson.kvitems(file, ""):
            question_types = item.get("types") or {}
            structural_type = question_types.get("structural")

            if structural_type not in candidates:
                continue

            image_id = str(item["imageId"])

            candidates[structural_type].append(
                {
                    "question_id": str(question_id),
                    "image_id": image_id,
                    "image": f"images/{image_id}.jpg",
                    "question": item["question"],
                    "answer": item["answer"],
                    "full_answer": item.get("fullAnswer"),
                    "types": question_types,
                }
            )

    return candidates


def select_unique_images(candidates, per_type, seed):
    selected = {
        structural_type: []
        for structural_type in STRUCTURAL_TYPES
    }
    used_image_ids = set()

    for structural_type in STRUCTURAL_TYPES:
        pool = candidates[structural_type]

        pool.sort(
            key=lambda item: stable_key(
                seed=seed,
                question_id=item["question_id"],
            )
        )

        for item in pool:
            if item["image_id"] in used_image_ids:
                continue

            selected[structural_type].append(item)
            used_image_ids.add(item["image_id"])

            if len(selected[structural_type]) == per_type:
                break

        if len(selected[structural_type]) != per_type:
            raise RuntimeError(
                f"Not enough unique images for {structural_type}: "
                f"{len(selected[structural_type])}/{per_type}"
            )

    return selected


def interleave(selected, per_type):
    records = []

    for index in range(per_type):
        for structural_type in STRUCTURAL_TYPES:
            records.append(selected[structural_type][index])

    return records


def write_jsonl(records, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def print_summary(name, records):
    structural_counts = Counter(
        item["types"]["structural"]
        for item in records
    )
    semantic_counts = Counter(
        item["types"]["semantic"]
        for item in records
    )
    unique_images = {
        item["image_id"]
        for item in records
    }

    print(f"\n{name}")
    print(f"  Samples: {len(records)}")
    print(f"  Unique images: {len(unique_images)}")
    print(f"  Structural types: {dict(structural_counts)}")
    print(f"  Semantic types: {dict(semantic_counts)}")
    print(f"  First sample: {records[0]}")


def main():
    args = parse_args()
    root = args.root.resolve()

    question_path = (
        root
        / "questions"
        / "val_balanced_questions.json"
    )
    subset_dir = root / "subsets"

    print(f"Loading: {question_path}")
    candidates = load_candidates(question_path)

    for structural_type in STRUCTURAL_TYPES:
        print(
            f"  {structural_type}: "
            f"{len(candidates[structural_type]):,} candidates"
        )

    selected = select_unique_images(
        candidates=candidates,
        per_type=20,
        seed=args.seed,
    )

    smoke_records = interleave(
        selected=selected,
        per_type=4,
    )
    debug_records = interleave(
        selected=selected,
        per_type=20,
    )

    smoke_path = subset_dir / "val_smoke_20.jsonl"
    debug_path = subset_dir / "val_debug_100.jsonl"

    write_jsonl(smoke_records, smoke_path)
    write_jsonl(debug_records, debug_path)

    metadata = {
        "source": question_path.name,
        "seed": args.seed,
        "unique_images": True,
        "selection": "balanced by structural type",
        "structural_types": list(STRUCTURAL_TYPES),
        "subsets": {
            smoke_path.name: len(smoke_records),
            debug_path.name: len(debug_records),
        },
    }

    metadata_path = subset_dir / "val_subset_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print_summary("Smoke subset", smoke_records)
    print_summary("Debug subset", debug_records)

    print("\nSaved files:")
    print(f"  {smoke_path}")
    print(f"  {debug_path}")
    print(f"  {metadata_path}")


if __name__ == "__main__":
    main()