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
)
from models.global_feature_fusion import (
    ConcatenationFusion,
    DescriptorConditionedFiLMGatedFusion,
    FeatureWiseGatedFusion,
)
from models.spatial_attention_encoder import (
    SpatialAttentionEncoder,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test spatial-statistical "
            "global feature fusion."
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
            "untrained_global_fusion_test.pt"
        ),
    )

    return parser.parse_args()


def parameter_count(
    model: torch.nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


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

    probability_map = (
        descriptor_result.probability_map
    )

    descriptors = descriptor_result.descriptors

    spatial_encoder = SpatialAttentionEncoder(
        map_size=32,
        output_dimension=128,
        dropout=0.1,
    )

    statistical_encoder = (
        AdaptiveGroupInteractionMLP(
            descriptor_dimension=16,
            group_count=4,
            descriptors_per_group=4,
            group_dimension=32,
            expansion_dimension=64,
            output_dimension=128,
            controller_dimension=32,
            dropout=0.1,
        )
    )

    proposed_fusion = (
        DescriptorConditionedFiLMGatedFusion(
            feature_dimension=128,
            film_hidden_dimension=128,
            gate_hidden_dimension=128,
            modulation_scale=0.5,
        )
    )

    concatenation_baseline = ConcatenationFusion(
        feature_dimension=128
    )

    gated_baseline = FeatureWiseGatedFusion(
        feature_dimension=128,
        gate_hidden_dimension=128,
    )

    print("Parameter counts:")
    print(
        "  Spatial encoder:",
        f"{parameter_count(spatial_encoder):,}",
    )
    print(
        "  Statistical encoder:",
        f"{parameter_count(statistical_encoder):,}",
    )
    print(
        "  Concatenation baseline:",
        f"{parameter_count(concatenation_baseline):,}",
    )
    print(
        "  Gated baseline:",
        f"{parameter_count(gated_baseline):,}",
    )
    print(
        "  Proposed fusion:",
        f"{parameter_count(proposed_fusion):,}",
    )

    spatial_encoder.train()
    statistical_encoder.train()
    proposed_fusion.train()

    spatial_feature = spatial_encoder(
        probability_map
    )

    statistical_output = statistical_encoder(
        descriptors
    )

    statistical_feature = (
        statistical_output.statistical_feature
    )

    training_output = proposed_fusion(
        spatial_feature,
        statistical_feature,
    )

    target = torch.linspace(
        -0.25,
        0.25,
        128,
    ).unsqueeze(0)

    test_loss = F.mse_loss(
        training_output.fused_feature,
        target,
    )

    test_loss.backward()

    spatial_gradient = gradient_norm(
        spatial_encoder.parameters()
    )

    statistical_gradient = gradient_norm(
        statistical_encoder.parameters()
    )

    fusion_gradient = gradient_norm(
        proposed_fusion.parameters()
    )

    film_gradient = gradient_norm(
        proposed_fusion.film_network.parameters()
    )

    gate_gradient = gradient_norm(
        proposed_fusion.gate_network.parameters()
    )

    interaction_gradient = gradient_norm(
        proposed_fusion
        .interaction_projection
        .parameters()
    )

    spatial_encoder.eval()
    statistical_encoder.eval()
    proposed_fusion.eval()
    concatenation_baseline.eval()
    gated_baseline.eval()

    with torch.no_grad():
        spatial_feature = spatial_encoder(
            probability_map
        )

        statistical_output = (
            statistical_encoder(descriptors)
        )

        statistical_feature = (
            statistical_output.statistical_feature
        )

        output = proposed_fusion(
            spatial_feature,
            statistical_feature,
        )

        concatenation_output = (
            concatenation_baseline(
                spatial_feature,
                statistical_feature,
            )
        )

        gated_output = gated_baseline(
            spatial_feature,
            statistical_feature,
        )

    film_identity_difference = (
        output.conditioned_spatial
        - output.normalized_spatial
    ).abs().max().item()

    gate_difference = (
        output.fusion_gate - 0.5
    ).abs().max().item()

    gamma_max = output.film_gamma.abs().max().item()
    beta_max = output.film_beta.abs().max().item()

    interaction_max = (
        output.interaction_feature
        .abs()
        .max()
        .item()
    )

    fused_feature = output.fused_feature

    has_nan = bool(
        torch.isnan(fused_feature).any().item()
    )

    has_inf = bool(
        torch.isinf(fused_feature).any().item()
    )

    print("\nBranch diagnostics:")
    print(
        "  Raw spatial norm:",
        f"{spatial_feature.norm().item():.8f}",
    )
    print(
        "  Raw statistical norm:",
        f"{statistical_feature.norm().item():.8f}",
    )
    print(
        "  Normalized spatial norm:",
        f"{output.normalized_spatial.norm().item():.8f}",
    )
    print(
        "  Normalized statistical norm:",
        f"{output.normalized_statistical.norm().item():.8f}",
    )

    print("\nInitialization diagnostics:")
    print(
        "  Maximum |gamma|:",
        f"{gamma_max:.8f}",
    )
    print(
        "  Maximum |beta|:",
        f"{beta_max:.8f}",
    )
    print(
        "  FiLM identity difference:",
        f"{film_identity_difference:.8f}",
    )
    print(
        "  Gate mean:",
        f"{output.fusion_gate.mean().item():.8f}",
    )
    print(
        "  Maximum gate difference from 0.5:",
        f"{gate_difference:.8f}",
    )
    print(
        "  Maximum interaction magnitude:",
        f"{interaction_max:.8f}",
    )

    print("\nFusion outputs:")
    print(
        "  Concatenation baseline:",
        tuple(concatenation_output.shape),
    )
    print(
        "  Gated baseline:",
        tuple(gated_output.shape),
    )
    print(
        "  Proposed fused feature:",
        tuple(fused_feature.shape),
    )
    print(
        "  Proposed feature mean:",
        f"{fused_feature.mean().item():.8f}",
    )
    print(
        "  Proposed feature std:",
        f"{fused_feature.std().item():.8f}",
    )
    print(
        "  Proposed feature norm:",
        f"{fused_feature.norm().item():.8f}",
    )
    print("  NaN:", has_nan)
    print("  Inf:", has_inf)

    print("\nGradient diagnostics:")
    print(
        "  Test loss:",
        f"{test_loss.item():.8f}",
    )
    print(
        "  Spatial gradient norm:",
        f"{spatial_gradient:.8f}",
    )
    print(
        "  Statistical gradient norm:",
        f"{statistical_gradient:.8f}",
    )
    print(
        "  Fusion gradient norm:",
        f"{fusion_gradient:.8f}",
    )
    print(
        "  FiLM gradient norm:",
        f"{film_gradient:.8f}",
    )
    print(
        "  Gate gradient norm:",
        f"{gate_gradient:.8f}",
    )
    print(
        "  Interaction gradient norm:",
        f"{interaction_gradient:.8f}",
    )

    checks = {
        "concatenation_parameters": (
            parameter_count(
                concatenation_baseline
            ) == 33_664
        ),
        "gated_parameters": (
            parameter_count(
                gated_baseline
            ) == 50_176
        ),
        "proposed_parameters": (
            parameter_count(
                proposed_fusion
            ) == 116_224
        ),
        "concatenation_shape": (
            concatenation_output.shape
            == (1, 128)
        ),
        "gated_shape": (
            gated_output.shape == (1, 128)
        ),
        "proposed_shape": (
            fused_feature.shape == (1, 128)
        ),
        "film_gamma_zero": gamma_max < 1e-7,
        "film_beta_zero": beta_max < 1e-7,
        "film_identity": (
            film_identity_difference < 1e-7
        ),
        "gate_half": gate_difference < 1e-7,
        "interaction_zero": (
            interaction_max < 1e-7
        ),
        "finite": not has_nan and not has_inf,
        "spatial_gradient": (
            spatial_gradient > 0.0
        ),
        "statistical_gradient": (
            statistical_gradient > 0.0
        ),
        "fusion_gradient": (
            fusion_gradient > 0.0
        ),
        "film_gradient": film_gradient > 0.0,
        "gate_gradient": gate_gradient > 0.0,
        "interaction_gradient": (
            interaction_gradient > 0.0
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

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "trained": False,
            "standardizer_fitted": bool(
                statistical_encoder
                .standardizer
                .fitted
                .item()
            ),
            "probability_map": (
                probability_map.cpu()
            ),
            "descriptors": descriptors.cpu(),
            "spatial_feature": (
                spatial_feature.cpu()
            ),
            "statistical_feature": (
                statistical_feature.cpu()
            ),
            "normalized_spatial": (
                output.normalized_spatial.cpu()
            ),
            "normalized_statistical": (
                output.normalized_statistical.cpu()
            ),
            "conditioned_spatial": (
                output.conditioned_spatial.cpu()
            ),
            "film_gamma": (
                output.film_gamma.cpu()
            ),
            "film_beta": output.film_beta.cpu(),
            "fusion_gate": (
                output.fusion_gate.cpu()
            ),
            "interaction_feature": (
                output.interaction_feature.cpu()
            ),
            "fused_feature": (
                fused_feature.cpu()
            ),
            "parameter_counts": {
                "concatenation": parameter_count(
                    concatenation_baseline
                ),
                "gated": parameter_count(
                    gated_baseline
                ),
                "proposed": parameter_count(
                    proposed_fusion
                ),
            },
        },
        output_path,
    )

    print("\nSaved untrained test artifact:")
    print(f"  {output_path}")

    if failed_checks:
        raise RuntimeError(
            "Global fusion test failed: "
            + ", ".join(failed_checks)
        )

    print("\nGlobal Feature Fusion test: PASS")
    print(
        "Note: feature quality is not evaluated "
        "because all trainable modules are untrained."
    )


if __name__ == "__main__":
    main()