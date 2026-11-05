import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


VLM_ROOT = Path(__file__).resolve().parents[1]

if str(VLM_ROOT) not in sys.path:
    sys.path.insert(0, str(VLM_ROOT))


from models.attention_descriptors import (
    AttentionDescriptorExtractor,
    DESCRIPTOR_GROUPS,
    DESCRIPTOR_NAMES,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test the 16 attention distribution "
            "descriptors."
        )
    )

    parser.add_argument(
        "--attention-map",
        type=str,
        default=(
            "/home/dm-magneto/dm_yunjin/vlm/"
            "outputs/attention_maps/candy/"
            "attention_map.npy"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "/home/dm-magneto/dm_yunjin/vlm/"
            "outputs/attention_maps/candy/"
            "attention_descriptors.json"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    attention_path = Path(args.attention_map)
    output_path = Path(args.output)

    if not attention_path.exists():
        raise FileNotFoundError(
            f"Attention map not found: {attention_path}"
        )

    attention_array = np.load(attention_path)

    extractor = AttentionDescriptorExtractor(
        map_size=32,
        top_p=0.05,
        peak_threshold=0.50,
        max_peak_count=16,
    )

    with torch.no_grad():
        result = extractor(
            torch.from_numpy(attention_array)
        )

    probability_map = result.probability_map
    descriptors = result.descriptors[0]
    descriptor_values = descriptors.cpu().tolist()

    map_sum = float(probability_map.sum().item())
    frequency_sum = float(
        descriptors[12:15].sum().item()
    )

    has_nan = bool(
        torch.isnan(descriptors).any().item()
    )

    has_inf = bool(
        torch.isinf(descriptors).any().item()
    )

    print("Attention descriptor summary:")
    print(
        "  Original map shape:",
        tuple(attention_array.shape),
    )
    print(
        "  Prepared map shape:",
        tuple(probability_map.shape),
    )
    print(f"  Prepared map sum: {map_sum:.8f}")
    print(
        "  Descriptor shape:",
        tuple(result.descriptors.shape),
    )
    print(
        "  Raw peak count:",
        int(result.peak_count_raw[0].item()),
    )
    print(f"  Frequency sum: {frequency_sum:.8f}")
    print("  NaN:", has_nan)
    print("  Inf:", has_inf)

    print("\nDescriptors:")

    for group_name, (start, end) in (
        DESCRIPTOR_GROUPS.items()
    ):
        print(f"\n  [{group_name}]")

        for index in range(start, end):
            print(
                f"    {DESCRIPTOR_NAMES[index]:<26} "
                f"{descriptor_values[index]:.8f}"
            )

    checks = {
        "shape": (
            result.descriptors.shape == (1, 16)
        ),
        "map_sum": abs(map_sum - 1.0) < 1e-5,
        "finite": not has_nan and not has_inf,
        "frequency_sum": (
            abs(frequency_sum - 1.0) < 1e-5
        ),
        "peak_count": (
            0
            <= int(result.peak_count_raw[0].item())
        ),
    }

    failed_checks = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "source_attention_map": str(
            attention_path
        ),
        "map_size": 32,
        "top_p": 0.05,
        "peak_threshold": 0.50,
        "max_peak_count": 16,
        "prepared_map_sum": map_sum,
        "raw_peak_count": int(
            result.peak_count_raw[0].item()
        ),
        "frequency_energy_sum": frequency_sum,
        "descriptors": {
            name: value
            for name, value in zip(
                DESCRIPTOR_NAMES,
                descriptor_values,
            )
        },
        "checks": checks,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nChecks:")

    for name, passed in checks.items():
        print(
            f"  {name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    print("\nSaved descriptor JSON:")
    print(f"  {output_path}")

    if failed_checks:
        raise RuntimeError(
            "Descriptor test failed: "
            + ", ".join(failed_checks)
        )

    print("\nAttention descriptor test: PASS")


if __name__ == "__main__":
    main()