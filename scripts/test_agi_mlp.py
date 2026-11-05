import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


VLM_ROOT = Path(__file__).resolve().parents[1]

if str(VLM_ROOT) not in sys.path:
    sys.path.insert(0, str(VLM_ROOT))


from models.agi_mlp import (
    AdaptiveGroupInteractionMLP,
)
from models.attention_descriptors import (
    AttentionDescriptorExtractor,
    DESCRIPTOR_GROUPS,
    DESCRIPTOR_NAMES,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test the AGI-MLP."
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
            "untrained_agi_mlp_test.pt"
        ),
    )

    return parser.parse_args()


def gradient_norm(
    parameters,
) -> float:
    total = 0.0

    for parameter in parameters:
        if parameter.grad is None:
            continue

        gradient = parameter.grad.detach().float()
        total += float(
            gradient.square().sum().item()
        )

    return total ** 0.5


def main():
    args = parse_args()

    torch.manual_seed(2026)

    attention_path = Path(args.attention_map)
    output_path = Path(args.output)

    if not attention_path.exists():
        raise FileNotFoundError(
            f"Attention map not found: {attention_path}"
        )

    attention_array = np.load(attention_path)

    descriptor_extractor = (
        AttentionDescriptorExtractor(
            map_size=32,
            top_p=0.05,
            peak_threshold=0.50,
            max_peak_count=16,
        )
    )

    with torch.no_grad():
        descriptor_result = descriptor_extractor(
            torch.from_numpy(attention_array)
        )

    descriptors = descriptor_result.descriptors

    model = AdaptiveGroupInteractionMLP(
        descriptor_dimension=16,
        group_count=4,
        descriptors_per_group=4,
        group_dimension=32,
        expansion_dimension=64,
        output_dimension=128,
        controller_dimension=32,
        dropout=0.1,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    model.train()

    training_output = model(descriptors)

    target = torch.linspace(
        -0.5,
        0.5,
        128,
    ).unsqueeze(0)

    test_loss = F.mse_loss(
        training_output.statistical_feature,
        target,
    )

    test_loss.backward()

    total_gradient = gradient_norm(
        model.parameters()
    )

    raw_gradient = gradient_norm(
        model.raw_projection.parameters()
    )

    controller_gradient = gradient_norm(
        model.controller.parameters()
    )

    group_gradient = gradient_norm(
        model.group_encoders.parameters()
    )

    model.eval()

    with torch.no_grad():
        output = model(descriptors)

    statistical_feature = (
        output.statistical_feature
    )

    mixing_matrix = output.mixing_matrix
    interaction_gates = (
        output.interaction_gates
    )

    row_sums = mixing_matrix.sum(dim=-1)

    diagonal = torch.diagonal(
        mixing_matrix,
        dim1=-2,
        dim2=-1,
    )

    has_nan = bool(
        torch.isnan(statistical_feature).any().item()
    )

    has_inf = bool(
        torch.isinf(statistical_feature).any().item()
    )

    print("AGI-MLP summary:")
    print(
        "  Descriptor input:",
        tuple(descriptors.shape),
    )
    print(
        "  Group features:",
        tuple(output.group_features.shape),
    )
    print(
        "  Interacted group features:",
        tuple(
            output.interacted_group_features.shape
        ),
    )
    print(
        "  Mixing matrix:",
        tuple(mixing_matrix.shape),
    )
    print(
        "  Statistical feature:",
        tuple(statistical_feature.shape),
    )
    print(
        "  Trainable parameters:",
        f"{parameter_count:,}",
    )
    print(
        "  Standardizer fitted:",
        bool(model.standardizer.fitted.item()),
    )

    print("\nInput descriptors:")

    descriptor_values = descriptors[0].tolist()

    for group_name, (start, end) in (
        DESCRIPTOR_GROUPS.items()
    ):
        print(f"\n  [{group_name}]")

        for index in range(start, end):
            print(
                f"    {DESCRIPTOR_NAMES[index]:<26} "
                f"{descriptor_values[index]:.8f}"
            )

    print("\nInitial mixing matrix:")

    for row in mixing_matrix[0]:
        formatted_row = " ".join(
            f"{value.item():.6f}"
            for value in row
        )
        print(f"  [{formatted_row}]")

    print("\nInitial interaction gates:")
    print(
        "  "
        + " ".join(
            f"{value.item():.6f}"
            for value in interaction_gates[0]
        )
    )

    print("\nOutput diagnostics:")
    print(
        "  Feature mean:",
        f"{statistical_feature.mean().item():.8f}",
    )
    print(
        "  Feature std:",
        f"{statistical_feature.std().item():.8f}",
    )
    print(
        "  Feature norm:",
        f"{statistical_feature.norm().item():.8f}",
    )
    print(
        "  Mixing row sums:",
        row_sums[0].tolist(),
    )
    print(
        "  Maximum diagonal value:",
        f"{diagonal.abs().max().item():.8f}",
    )
    print(
        "  Test loss:",
        f"{test_loss.item():.8f}",
    )
    print(
        "  Total gradient norm:",
        f"{total_gradient:.8f}",
    )
    print(
        "  Raw path gradient norm:",
        f"{raw_gradient:.8f}",
    )
    print(
        "  Controller gradient norm:",
        f"{controller_gradient:.8f}",
    )
    print(
        "  Group encoder gradient norm:",
        f"{group_gradient:.8f}",
    )
    print("  NaN:", has_nan)
    print("  Inf:", has_inf)

    checks = {
        "group_shape": (
            output.group_features.shape
            == (1, 4, 32)
        ),
        "interaction_shape": (
            output.interacted_group_features.shape
            == (1, 4, 32)
        ),
        "mixing_shape": (
            mixing_matrix.shape == (1, 4, 4)
        ),
        "output_shape": (
            statistical_feature.shape == (1, 128)
        ),
        "parameter_count": (
            parameter_count == 31_668
        ),
        "row_normalization": bool(
            torch.allclose(
                row_sums,
                torch.ones_like(row_sums),
                atol=1e-6,
            )
        ),
        "zero_diagonal": (
            diagonal.abs().max().item() < 1e-6
        ),
        "valid_gates": bool(
            (
                (interaction_gates > 0.0)
                & (interaction_gates < 1.0)
            ).all().item()
        ),
        "finite": not has_nan and not has_inf,
        "total_gradient": total_gradient > 0.0,
        "raw_gradient": raw_gradient > 0.0,
        "controller_gradient": (
            controller_gradient > 0.0
        ),
        "group_gradient": group_gradient > 0.0,
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

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "trained": False,
            "standardizer_fitted": bool(
                model.standardizer.fitted.item()
            ),
            "descriptor_names": DESCRIPTOR_NAMES,
            "descriptors": descriptors.cpu(),
            "standardized_descriptors": (
                output.standardized_descriptors.cpu()
            ),
            "group_features": (
                output.group_features.cpu()
            ),
            "interacted_group_features": (
                output.interacted_group_features.cpu()
            ),
            "mixing_matrix": mixing_matrix.cpu(),
            "interaction_gates": (
                interaction_gates.cpu()
            ),
            "statistical_feature": (
                statistical_feature.cpu()
            ),
            "parameter_count": parameter_count,
        },
        output_path,
    )

    print("\nSaved untrained test artifact:")
    print(f"  {output_path}")

    if failed_checks:
        raise RuntimeError(
            "AGI-MLP test failed: "
            + ", ".join(failed_checks)
        )

    print("\nAGI-MLP test: PASS")
    print(
        "Note: descriptor statistics are not fitted "
        "and feature quality is not evaluated."
    )


if __name__ == "__main__":
    main()