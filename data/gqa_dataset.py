import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


REQUIRED_FIELDS = (
    "question_id",
    "image_id",
    "image",
    "question",
    "answer",
    "types",
)



def normalize_gqa_answer(answer):
    """
    Apply conservative GQA answer
    normalization.
    """

    if not isinstance(answer, str):
        answer = str(answer)

    normalized = unicodedata.normalize(
        "NFKC",
        answer,
    )

    normalized = normalized.strip().lower()

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    if not normalized:
        raise ValueError(
            "GQA answer must not be empty."
        )

    return normalized


class GQADataset(Dataset):
    def __init__(
        self,
        manifest_path,
        data_root=None,
        verify_images=False,
    ):
        self.manifest_path = Path(manifest_path).resolve()

        if data_root is None:
            self.data_root = self.manifest_path.parents[1]
        else:
            self.data_root = Path(data_root).resolve()

        self.records = self._load_records()

        if verify_images:
            self._verify_images()

    def _load_records(self):
        records = []

        with self.manifest_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                record = json.loads(line)

                missing_fields = [
                    field
                    for field in REQUIRED_FIELDS
                    if field not in record
                ]

                if missing_fields:
                    raise ValueError(
                        f"{self.manifest_path}:{line_number} "
                        f"missing fields: {missing_fields}"
                    )

                question = record["question"]

                if (
                    not isinstance(question, str)
                    or not question.strip()
                ):
                    raise ValueError(
                        f"{self.manifest_path}:"
                        f"{line_number} contains "
                        "an empty question."
                    )

                normalized_answer = (
                    normalize_gqa_answer(
                        record["answer"]
                    )
                )

                stored_normalized_answer = (
                    record.get(
                        "normalized_answer"
                    )
                )

                if (
                    stored_normalized_answer
                    is not None
                    and stored_normalized_answer
                    != normalized_answer
                ):
                    raise ValueError(
                        f"{self.manifest_path}:"
                        f"{line_number} has an "
                        "inconsistent normalized "
                        "answer."
                    )

                record[
                    "normalized_answer"
                ] = normalized_answer

                records.append(record)

        if not records:
            raise ValueError(
                f"No samples found in {self.manifest_path}"
            )

        return records

    def _resolve_image_path(self, record):
        return (self.data_root / record["image"]).resolve()

    def _verify_images(self):
        missing_paths = []

        for record in self.records:
            image_path = self._resolve_image_path(record)

            if not image_path.is_file():
                missing_paths.append(str(image_path))

                if len(missing_paths) >= 10:
                    break

        if missing_paths:
            raise FileNotFoundError(
                "Missing GQA images:\n"
                + "\n".join(missing_paths)
            )

    def summary(self):
        structural_types = Counter(
            record["types"]["structural"]
            for record in self.records
        )
        semantic_types = Counter(
            record["types"]["semantic"]
            for record in self.records
        )
        unique_images = {
            record["image_id"]
            for record in self.records
        }

        return {
            "samples": len(self.records),
            "unique_images": len(unique_images),
            "structural_types": dict(structural_types),
            "semantic_types": dict(semantic_types),
        }

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        sample = dict(record)

        sample["index"] = index
        sample["image_path"] = str(
            self._resolve_image_path(record)
        )

        return sample


def collate_gqa_samples(samples):
    return samples


def build_gqa_dataloader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    seed=2026,
    epoch=0,
):
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    if num_workers < 0:
        raise ValueError(
            "num_workers cannot be negative."
        )

    if epoch < 0:
        raise ValueError(
            "epoch cannot be negative."
        )

    generator = None

    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(
            int(seed) + int(epoch)
        )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_gqa_samples,
        pin_memory=False,
        generator=generator,
        drop_last=False,
    )
