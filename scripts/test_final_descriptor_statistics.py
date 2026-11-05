import argparse
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import numpy as np
import torch
import torch.nn.functional as F

from vlm.models.global_token_encoder import (
    GlobalTokenEncoder,
)
from vlm.utils.descriptor_statistics import (
    load_descriptor_statistics,
)


EXPECTED_DESCRIPTOR_NAMES = (
    "peak_value",
    "top_5_percent_mass",
    "spatial_entropy",
    "hoyer_sparsity",
    "centroid_x",
    "centroid_y",
    "right_left_balance",
    "bottom_top_balance",
    "spread_x",
    "spread_y",
    "eccentricity",
    "peak_count",
    "low_frequency_energy",
    "mid_frequency_energy",
    "high_frequency_energy",
    "spectral_entropy",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Apply the final GQA descriptor statistics "
            "to GlobalTokenEncoder and validate them."
        )
    )

    parser.add_argument(
        "--statistics",
        type=str,
        default=(
            "/home/dm-magneto/dm_yunjin/vlm/configs/"
            "gqa_descriptor_statistics_5000_seed2026.json"
        ),
    )

    parser.add_argument(
        "--attention-map",
        type=str,
        default=(
            "/home/dm-magneto/dm_yunjin/vlm/outputs/"
            "attention_maps/candy/attention_map.npy"
        ),
    )

    return parser.parse_args()


def parameter_count(module):
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def gradient_norm(module):
    squared_norm = 0.0

    for parameter in module.parameters():
        if parameter.grad is None:
            continue

        squared_norm += (
            parameter.grad.detach().float().norm().item()
            ** 2
        )

    return squared_norm ** 0.5


def tensors_are_finite(tensors):
    return all(
        bool(torch.isfinite(tensor).all())
        for tensor in tensors
    )


def main():
    args = parse_args()

    torch.manual_seed(2026)

    statistics_path = Path(args.statistics)
    attention_map_path = Path(args.attention_map)

    if not statistics_path.is_file():
        raise FileNotFoundError(
            f"Statistics not found: {statistics_path}"
        )

    if not attention_map_path.is_file():
        raise FileNotFoundError(
            f"Attention map not found: "
            f"{attention_map_path}"
        )

    statistics = load_descriptor_statistics(
        statistics_path,
        expected_descriptor_names=(
            EXPECTED_DESCRIPTOR_NAMES
        ),
    )

    mean = statistics["mean"].float()
    std = statistics["std"].float()
    raw_std = statistics["raw_std"].float()
    metadata = statistics["metadata"]

    attention_array = np.load(
        attention_map_path
    )

    attention_map = torch.from_numpy(
        attention_array
    ).float()

    model = GlobalTokenEncoder(
        map_size=32,
        descriptor_dimension=16,
        feature_dimension=128,
        llm_hidden_dimension=3584,
        top_p=0.05,
        peak_threshold=0.50,
        max_peak_count=16,
        dropout=0.1,
        modulation_scale=0.5,
        initial_token_scale=0.35,
    )

    fitted_before = (
        model.descriptor_statistics_fitted
    )

    model.set_descriptor_statistics(
        mean=mean,
        std=std,
    )

    fitted_after = (
        model.descriptor_statistics_fitted
    )

    model.eval()

    with torch.no_grad():
        output = model(attention_map)
        repeated_output = model(attention_map)

    descriptors = (
        output.descriptor_output.descriptors
    )

    standardized_descriptors = (
        output.statistical_output
        .standardized_descriptors
    )

    expected_standardized = (
        descriptors - mean.view(1, -1)
    ) / std.view(1, -1)

    standardization_error = (
        standardized_descriptors
        - expected_standardized
    ).abs().max().item()

    probability_map = (
        output.descriptor_output.probability_map
    )

    global_token = output.global_token

    normalized_feature = (
        output.projection_output.normalized_feature
    )

    token_scale = (
        output.projection_output
        .token_scale.detach().float().item()
    )

    normalized_rms = (
        normalized_feature.float()
        .pow(2)
        .mean()
        .sqrt()
        .item()
    )

    global_token_norm = (
        global_token.float().norm().item()
    )

    expected_token_norm = (
        token_scale * math.sqrt(3584)
    )

    token_norm_error = abs(
        global_token_norm
        - expected_token_norm
    )

    deterministic_error = (
        output.global_token
        - repeated_output.global_token
    ).abs().max().item()

    finite_tensors = [
        mean,
        std,
        raw_std,
        probability_map,
        descriptors,
        standardized_descriptors,
        output.spatial_feature,
        output.statistical_output.statistical_feature,
        output.statistical_output.group_features,
        output.statistical_output.interacted_group_features,
        output.statistical_output.mixing_matrix,
        output.fusion_output.fused_feature,
        output.fusion_output.film_gamma,
        output.fusion_output.film_beta,
        output.fusion_output.fusion_gate,
        output.fusion_output.interaction_feature,
        output.projection_output.projected_feature,
        normalized_feature,
        global_token,
    ]

    model.train()
    model.zero_grad(set_to_none=True)

    training_output = model(attention_map)

    target = torch.linspace(
        -0.1,
        0.1,
        3584,
        dtype=training_output.global_token.dtype,
    ).view(1, 1, 3584)

    loss = F.mse_loss(
        training_output.global_token.float(),
        target.float(),
    )

    loss.backward()

    gradient_norms = {
        "spatial": gradient_norm(
            model.spatial_encoder
        ),
        "statistical": gradient_norm(
            model.statistical_encoder
        ),
        "fusion": gradient_norm(
            model.fusion
        ),
        "projector": gradient_norm(
            model.projector
        ),
    }

    checks = {
        "statistics_count": (
            statistics["count"] == 5000
        ),
        "descriptor_names": (
            tuple(statistics["descriptor_names"])
            == EXPECTED_DESCRIPTOR_NAMES
        ),
        "statistics_shape": (
            tuple(mean.shape) == (16,)
            and tuple(std.shape) == (16,)
        ),
        "statistics_finite": (
            tensors_are_finite(
                [mean, std, raw_std]
            )
        ),
        "positive_standard_deviation": bool(
            (std > 0).all()
        ),
        "no_clamped_standard_deviation": (
            torch.allclose(
                std,
                raw_std,
                atol=0.0,
                rtol=0.0,
            )
        ),
        "training_split": (
            metadata.get("dataset") == "GQA"
            and metadata.get("split") == "train"
        ),
        "metadata_sample_count": (
            metadata.get("sample_count") == 5000
        ),
        "metadata_map_size": (
            metadata.get("map_size") == 32
        ),
        "metadata_top_p": (
            abs(
                float(metadata.get("top_p"))
                - 0.05
            ) < 1e-12
        ),
        "metadata_peak_threshold": (
            abs(
                float(
                    metadata.get(
                        "peak_threshold"
                    )
                )
                - 0.50
            ) < 1e-12
        ),
        "metadata_max_peak_count": (
            metadata.get("max_peak_count") == 16
        ),
        "not_fitted_before": (
            fitted_before is False
        ),
        "fitted_after": (
            fitted_after is True
        ),
        "probability_map_shape": (
            tuple(probability_map.shape)
            == (1, 1, 32, 32)
        ),
        "probability_map_sum": (
            abs(
                probability_map.float()
                .sum()
                .item()
                - 1.0
            )
            < 1e-5
        ),
        "descriptor_shape": (
            tuple(descriptors.shape)
            == (1, 16)
        ),
        "standardized_shape": (
            tuple(
                standardized_descriptors.shape
            )
            == (1, 16)
        ),
        "standardization_exact": (
            standardization_error < 1e-5
        ),
        "global_token_shape": (
            tuple(global_token.shape)
            == (1, 1, 3584)
        ),
        "normalized_rms": (
            abs(normalized_rms - 1.0)
            < 1e-3
        ),
        "token_norm": (
            token_norm_error < 1e-2
        ),
        "deterministic_eval": (
            deterministic_error < 1e-7
        ),
        "finite_outputs": (
            tensors_are_finite(
                finite_tensors
            )
        ),
        "finite_loss": (
            math.isfinite(loss.item())
        ),
        "spatial_gradient": (
            gradient_norms["spatial"] > 0.0
        ),
        "statistical_gradient": (
            gradient_norms["statistical"] > 0.0
        ),
        "fusion_gradient": (
            gradient_norms["fusion"] > 0.0
        ),
        "projector_gradient": (
            gradient_norms["projector"] > 0.0
        ),
    }

    print(
        "Final Descriptor Statistics summary:"
    )
    print(
        f"  Statistics: {statistics_path}"
    )
    print(
        f"  Calibration samples: "
        f"{statistics['count']}"
    )
    print(
        f"  Descriptor dimension: "
        f"{mean.numel()}"
    )
    print(
        f"  Mean range: "
        f"{mean.min().item():.8f} - "
        f"{mean.max().item():.8f}"
    )
    print(
        f"  Std range: "
        f"{std.min().item():.8f} - "
        f"{std.max().item():.8f}"
    )
    print(
        f"  Clamped descriptors: "
        f"{int((std != raw_std).sum().item())}"
    )
    print(
        f"  Encoder fitted before: "
        f"{fitted_before}"
    )
    print(
        f"  Encoder fitted after: "
        f"{fitted_after}"
    )
    print(
        f"  Trainable parameters: "
        f"{parameter_count(model):,}"
    )

    print(
        "\nStandardized Candy descriptors:"
    )

    for index, name in enumerate(
        EXPECTED_DESCRIPTOR_NAMES
    ):
        raw_value = (
            descriptors[0, index]
            .float()
            .item()
        )
        standardized_value = (
            standardized_descriptors[0, index]
            .float()
            .item()
        )

        print(
            f"  {name:27s} "
            f"raw={raw_value: .8f} "
            f"z={standardized_value: .6f}"
        )

    print(
        "\nGlobal Token diagnostics:"
    )
    print(
        f"  Probability map: "
        f"{tuple(probability_map.shape)}"
    )
    print(
        f"  Descriptor shape: "
        f"{tuple(descriptors.shape)}"
    )
    print(
        f"  Standardized shape: "
        f"{tuple(standardized_descriptors.shape)}"
    )
    print(
        f"  Global Token: "
        f"{tuple(global_token.shape)}"
    )
    print(
        f"  Standardization max error: "
        f"{standardization_error:.10f}"
    )
    print(
        f"  Normalized feature RMS: "
        f"{normalized_rms:.8f}"
    )
    print(
        f"  Token scale: "
        f"{token_scale:.8f}"
    )
    print(
        f"  Token norm: "
        f"{global_token_norm:.8f}"
    )
    print(
        f"  Expected token norm: "
        f"{expected_token_norm:.8f}"
    )
    print(
        f"  Token norm error: "
        f"{token_norm_error:.8f}"
    )
    print(
        f"  Eval repeat max error: "
        f"{deterministic_error:.10f}"
    )

    print(
        "\nGradient diagnostics:"
    )
    print(
        f"  Test loss: "
        f"{loss.item():.8f}"
    )

    for name, norm in gradient_norms.items():
        print(
            f"  {name} gradient norm: "
            f"{norm:.8f}"
        )

    print("\nChecks:")

    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    if not all(checks.values()):
        failed_checks = [
            name
            for name, passed in checks.items()
            if not passed
        ]

        raise RuntimeError(
            "Final descriptor statistics "
            "validation failed: "
            + ", ".join(failed_checks)
        )

    print(
        "\nFinal Descriptor Statistics "
        "application test: PASS"
    )
    print(
        "Step 22 complete."
    )
    print(
        "Note: this validates statistics "
        "application and numerical behavior, "
        "not task performance."
    )


if __name__ == "__main__":
    main()