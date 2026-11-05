import argparse
import sys
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from vlm.data.gqa_dataset import (
    GQADataset,
    build_gqa_dataloader,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test the GQA Dataset and DataLoader."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "/home/dm-magneto/dm_yunjin/"
            "datasets/gqa/subsets/val_smoke_20.jsonl"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "/home/dm-magneto/dm_yunjin/datasets/gqa"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = GQADataset(
        manifest_path=args.manifest,
        data_root=args.data_root,
        verify_images=True,
    )

    dataloader = build_gqa_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print("Dataset summary:")
    for key, value in dataset.summary().items():
        print(f"  {key}: {value}")

    first_sample = dataset[0]

    with Image.open(first_sample["image_path"]) as image:
        print("\nFirst image:")
        print(f"  Path: {first_sample['image_path']}")
        print(f"  Size: {image.size}")
        print(f"  Mode: {image.mode}")

    first_batch = next(iter(dataloader))

    print(f"\nFirst batch size: {len(first_batch)}")

    for sample in first_batch:
        print(
            f"  index={sample['index']}, "
            f"question_id={sample['question_id']}, "
            f"image_id={sample['image_id']}"
        )
        print(f"    question: {sample['question']}")
        print(f"    answer: {sample['answer']}")
        print(
            f"    type: "
            f"{sample['types']['structural']} / "
            f"{sample['types']['semantic']} / "
            f"{sample['types']['detailed']}"
        )
        print(f"    image exists: {Path(sample['image_path']).is_file()}")

    print("\nGQA Dataset/DataLoader test: PASS")


if __name__ == "__main__":
    main()