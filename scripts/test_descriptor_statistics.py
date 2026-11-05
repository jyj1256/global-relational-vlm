import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from vlm.models.attention_descriptors import (
    DESCRIPTOR_NAMES,
)
from vlm.models.global_token_encoder import (
    GlobalTokenEncoder,
)
from vlm.utils.descriptor_statistics import (
    OnlineDescriptorStatistics,
    load_descriptor_statistics,
    save_descriptor_statistics,
)


def main():
    torch.manual_seed(2026)

    output_path = Path(
        "/home/dm-magneto/dm_yunjin/"
        "vlm/outputs/logs/"
        "test_descriptor_statistics.json"
    )

    row_values = torch.arange(
        8,
        dtype=torch.float64,
    ).unsqueeze(1)

    feature_scales = torch.linspace(
        0.1,
        1.6,
        16,
        dtype=torch.float64,
    ).unsqueeze(0)

    feature_offsets = torch.linspace(
        -0.5,
        0.5,
        16,
        dtype=torch.float64,
    ).unsqueeze(0)

    descriptor_data = (
        row_values * feature_scales
        + feature_offsets
    )

    accumulator = OnlineDescriptorStatistics(
        descriptor_names=DESCRIPTOR_NAMES,
        eps=1e-6,
    )

    # Test multiple online update batch sizes.
    accumulator.update(
        descriptor_data[0:3]
    )

    accumulator.update(
        descriptor_data[3:5]
    )

    accumulator.update(
        descriptor_data[5:8]
    )

    statistics = accumulator.finalize()

    expected_mean = descriptor_data.mean(
        dim=0
    ).float()

    expected_std = descriptor_data.std(
        dim=0,
        unbiased=False,
    ).float()

    save_descriptor_statistics(
        output_path=output_path,
        statistics=statistics,
        metadata={
            "test": True,
            "source": "synthetic",
        },
    )

    loaded = load_descriptor_statistics(
        input_path=output_path,
        expected_descriptor_names=(
            DESCRIPTOR_NAMES
        ),
    )

    standardized = (
        descriptor_data.float()
        - loaded["mean"]
    ) / loaded["std"]

    standardized_mean = standardized.mean(
        dim=0
    )

    standardized_std = standardized.std(
        dim=0,
        unbiased=False,
    )

    global_encoder = GlobalTokenEncoder()

    global_encoder.set_descriptor_statistics(
        mean=loaded["mean"],
        std=loaded["std"],
    )

    stored_mean = (
        global_encoder
        .statistical_encoder
        .standardizer
        .descriptor_mean
        .cpu()
    )

    stored_std = (
        global_encoder
        .statistical_encoder
        .standardizer
        .descriptor_std
        .cpu()
    )

    print("Descriptor Statistics summary:")
    print("  Count:", loaded["count"])
    print(
        "  Dimension:",
        len(loaded["descriptor_names"]),
    )
    print(
        "  Mean range:",
        f"{loaded['mean'].min().item():.8f} - "
        f"{loaded['mean'].max().item():.8f}",
    )
    print(
        "  Std range:",
        f"{loaded['std'].min().item():.8f} - "
        f"{loaded['std'].max().item():.8f}",
    )
    print(
        "  Standardized mean max abs:",
        f"{standardized_mean.abs().max().item():.8f}",
    )
    print(
        "  Standardized std error max:",
        f"{(standardized_std - 1.0).abs().max().item():.8f}",
    )
    print(
        "  Global encoder fitted:",
        global_encoder
        .descriptor_statistics_fitted,
    )

    checks = {
        "count": loaded["count"] == 8,
        "dimension": (
            len(loaded["descriptor_names"]) == 16
        ),
        "online_mean": bool(
            torch.allclose(
                statistics["mean"],
                expected_mean,
                atol=1e-6,
            )
        ),
        "online_std": bool(
            torch.allclose(
                statistics["std"],
                expected_std,
                atol=1e-6,
            )
        ),
        "saved_mean": bool(
            torch.allclose(
                loaded["mean"],
                expected_mean,
                atol=1e-6,
            )
        ),
        "saved_std": bool(
            torch.allclose(
                loaded["std"],
                expected_std,
                atol=1e-6,
            )
        ),
        "standardized_mean": bool(
            torch.allclose(
                standardized_mean,
                torch.zeros_like(
                    standardized_mean
                ),
                atol=1e-6,
            )
        ),
        "standardized_std": bool(
            torch.allclose(
                standardized_std,
                torch.ones_like(
                    standardized_std
                ),
                atol=1e-6,
            )
        ),
        "encoder_fitted": (
            global_encoder
            .descriptor_statistics_fitted
        ),
        "encoder_mean": bool(
            torch.allclose(
                stored_mean,
                loaded["mean"],
                atol=1e-7,
            )
        ),
        "encoder_std": bool(
            torch.allclose(
                stored_std,
                loaded["std"],
                atol=1e-7,
            )
        ),
    }

    failed_checks = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    print("\nChecks:")

    for name, passed in checks.items():
        print(
            f"  {name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    print("\nSaved test statistics:")
    print(f"  {output_path}")

    if failed_checks:
        raise RuntimeError(
            "Descriptor statistics test failed: "
            + ", ".join(failed_checks)
        )

    print("\nDescriptor Statistics test: PASS")


if __name__ == "__main__":
    main()