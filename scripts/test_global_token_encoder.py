import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from vlm.models.global_token_encoder import (
    GlobalTokenEncoder,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test the integrated Global Token Encoder."
        )
    )

    parser.add_argument(
        "--attention-map",
        type=Path,
        default=Path(
            "/home/dm-magneto/dm_yunjin/"
            "vlm/outputs/attention_maps/candy/"
            "attention_map.npy"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/dm-magneto/dm_yunjin/"
            "vlm/outputs/attention_maps/candy/"
            "untrained_global_encoder_test.pt"
        ),
    )

    return parser.parse_args()


def parameter_count(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def gradient_norm(parameters):
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

    if not args.attention_map.exists():
        raise FileNotFoundError(
            f"Attention map not found: "
            f"{args.attention_map}"
        )

    attention_array = np.load(
        args.attention_map
    )

    attention_map = torch.from_numpy(
        attention_array
    )

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

    counts = {
        "descriptor_extractor": parameter_count(
            model.descriptor_extractor
        ),
        "spatial_encoder": parameter_count(
            model.spatial_encoder
        ),
        "statistical_encoder": parameter_count(
            model.statistical_encoder
        ),
        "fusion": parameter_count(
            model.fusion
        ),
        "projector": parameter_count(
            model.projector
        ),
        "total": parameter_count(model),
    }

    print("GlobalTokenEncoder parameters:")

    for name, count in counts.items():
        print(f"  {name}: {count:,}")

    model.train()

    training_output = model(attention_map)

    target = torch.linspace(
        -0.1,
        0.1,
        3584,
    ).view(1, 1, 3584)

    test_loss = F.mse_loss(
        training_output.global_token,
        target,
    )

    test_loss.backward()

    gradients = {
        "spatial": gradient_norm(
            model.spatial_encoder.parameters()
        ),
        "statistical": gradient_norm(
            model.statistical_encoder.parameters()
        ),
        "fusion": gradient_norm(
            model.fusion.parameters()
        ),
        "projector": gradient_norm(
            model.projector.parameters()
        ),
    }

    model.eval()

    with torch.no_grad():
        output = model(attention_map)

    global_token = output.global_token
    descriptors = (
        output.descriptor_output.descriptors
    )
    probability_map = (
        output.descriptor_output.probability_map
    )
    mixing_matrix = (
        output.statistical_output.mixing_matrix
    )
    interaction_gates = (
        output.statistical_output
        .interaction_gates
    )
    fusion_gate = (
        output.fusion_output.fusion_gate
    )

    row_sums = mixing_matrix.sum(dim=-1)

    token_scale = float(
        output.projection_output
        .token_scale
        .item()
    )

    token_norm = float(
        global_token.norm().item()
    )

    expected_norm = (
        token_scale * math.sqrt(3584.0)
    )

    norm_error = abs(
        token_norm - expected_norm
    )

    has_nan = bool(
        torch.isnan(global_token).any().item()
    )

    has_inf = bool(
        torch.isinf(global_token).any().item()
    )

    print("\nShape summary:")
    print(
        "  Probability map:",
        tuple(probability_map.shape),
    )
    print(
        "  Descriptors:",
        tuple(descriptors.shape),
    )
    print(
        "  Spatial feature:",
        tuple(output.spatial_feature.shape),
    )
    print(
        "  Statistical feature:",
        tuple(
            output.statistical_output
            .statistical_feature
            .shape
        ),
    )
    print(
        "  Fused feature:",
        tuple(
            output.fusion_output
            .fused_feature
            .shape
        ),
    )
    print(
        "  Global token:",
        tuple(global_token.shape),
    )

    print("\nInteraction diagnostics:")
    print(
        "  Mixing matrix row sums:",
        row_sums[0].tolist(),
    )
    print(
        "  AGI interaction gate mean:",
        f"{interaction_gates.mean().item():.8f}",
    )
    print(
        "  Fusion gate mean:",
        f"{fusion_gate.mean().item():.8f}",
    )
    print(
        "  Descriptor statistics fitted:",
        model.descriptor_statistics_fitted,
    )

    print("\nGlobal Token diagnostics:")
    print(
        "  Token scale:",
        f"{token_scale:.8f}",
    )
    print(
        "  Token norm:",
        f"{token_norm:.8f}",
    )
    print(
        "  Expected norm:",
        f"{expected_norm:.8f}",
    )
    print(
        "  Norm error:",
        f"{norm_error:.8f}",
    )
    print("  NaN:", has_nan)
    print("  Inf:", has_inf)

    print("\nGradient diagnostics:")
    print(
        "  Test loss:",
        f"{test_loss.item():.8f}",
    )

    for name, value in gradients.items():
        print(
            f"  {name} gradient norm: "
            f"{value:.8f}"
        )

    checks = {
        "descriptor_parameters": (
            counts["descriptor_extractor"] == 0
        ),
        "spatial_parameters": (
            counts["spatial_encoder"] == 274_656
        ),
        "statistical_parameters": (
            counts["statistical_encoder"]
            == 31_668
        ),
        "fusion_parameters": (
            counts["fusion"] == 116_224
        ),
        "projector_parameters": (
            counts["projector"] == 465_921
        ),
        "total_parameters": (
            counts["total"] == 888_469
        ),
        "probability_shape": (
            probability_map.shape
            == (1, 1, 32, 32)
        ),
        "descriptor_shape": (
            descriptors.shape == (1, 16)
        ),
        "spatial_shape": (
            output.spatial_feature.shape
            == (1, 128)
        ),
        "statistical_shape": (
            output.statistical_output
            .statistical_feature
            .shape
            == (1, 128)
        ),
        "fusion_shape": (
            output.fusion_output
            .fused_feature
            .shape
            == (1, 128)
        ),
        "global_token_shape": (
            global_token.shape
            == (1, 1, 3584)
        ),
        "mixing_rows": bool(
            torch.allclose(
                row_sums,
                torch.ones_like(row_sums),
                atol=1e-6,
            )
        ),
        "token_norm": norm_error < 1e-3,
        "finite": not has_nan and not has_inf,
        "spatial_gradient": (
            gradients["spatial"] > 0.0
        ),
        "statistical_gradient": (
            gradients["statistical"] > 0.0
        ),
        "fusion_gradient": (
            gradients["fusion"] > 0.0
        ),
        "projector_gradient": (
            gradients["projector"] > 0.0
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

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "trained": False,
            "descriptor_statistics_fitted": (
                model.descriptor_statistics_fitted
            ),
            "global_token": global_token.cpu(),
            "probability_map": (
                probability_map.cpu()
            ),
            "descriptors": descriptors.cpu(),
            "spatial_feature": (
                output.spatial_feature.cpu()
            ),
            "statistical_feature": (
                output.statistical_output
                .statistical_feature
                .cpu()
            ),
            "mixing_matrix": (
                mixing_matrix.cpu()
            ),
            "interaction_gates": (
                interaction_gates.cpu()
            ),
            "fusion_gate": fusion_gate.cpu(),
            "fused_feature": (
                output.fusion_output
                .fused_feature
                .cpu()
            ),
            "parameter_counts": counts,
        },
        args.output,
    )

    print("\nSaved wrapper test artifact:")
    print(f"  {args.output}")

    if failed_checks:
        raise RuntimeError(
            "GlobalTokenEncoder test failed: "
            + ", ".join(failed_checks)
        )

    print("\nGlobalTokenEncoder test: PASS")
    print(
        "Note: descriptor statistics are not fitted "
        "and the encoder is untrained."
    )


if __name__ == "__main__":
    main()