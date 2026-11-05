import argparse
import math
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
    DescriptorConditionedFiLMGatedFusion,
)
from models.global_token_projector import (
    GlobalTokenProjector,
)
from models.spatial_attention_encoder import (
    SpatialAttentionEncoder,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test the complete untrained Global "
            "Token Encoder path."
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
            "untrained_global_token_test.pt"
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

    fusion = (
        DescriptorConditionedFiLMGatedFusion(
            feature_dimension=128,
            film_hidden_dimension=128,
            gate_hidden_dimension=128,
            modulation_scale=0.5,
        )
    )

    projector = GlobalTokenProjector(
        input_dimension=128,
        llm_hidden_dimension=3584,
        initial_token_scale=0.35,
        rms_norm_eps=1e-6,
    )

    spatial_parameters = parameter_count(
        spatial_encoder
    )

    statistical_parameters = parameter_count(
        statistical_encoder
    )

    fusion_parameters = parameter_count(fusion)
    projector_parameters = parameter_count(
        projector
    )

    total_parameters = (
        spatial_parameters
        + statistical_parameters
        + fusion_parameters
        + projector_parameters
    )

    print("Global Token Encoder parameters:")
    print(
        "  Spatial encoder:",
        f"{spatial_parameters:,}",
    )
    print(
        "  Statistical encoder:",
        f"{statistical_parameters:,}",
    )
    print(
        "  Fusion:",
        f"{fusion_parameters:,}",
    )
    print(
        "  Global projector:",
        f"{projector_parameters:,}",
    )
    print(
        "  Total:",
        f"{total_parameters:,}",
    )

    spatial_encoder.train()
    statistical_encoder.train()
    fusion.train()
    projector.train()

    spatial_feature = spatial_encoder(
        probability_map
    )

    statistical_output = statistical_encoder(
        descriptors
    )

    fusion_output = fusion(
        spatial_feature,
        statistical_output.statistical_feature,
    )

    projection_output = projector(
        fusion_output.fused_feature
    )

    target = torch.linspace(
        -0.1,
        0.1,
        3584,
    ).view(1, 1, 3584)

    test_loss = F.mse_loss(
        projection_output.global_token,
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
        fusion.parameters()
    )

    projector_gradient = gradient_norm(
        projector.parameters()
    )

    scale_gradient = float(
        projector
        .log_token_scale
        .grad
        .detach()
        .abs()
        .item()
    )

    spatial_encoder.eval()
    statistical_encoder.eval()
    fusion.eval()
    projector.eval()

    with torch.no_grad():
        spatial_feature = spatial_encoder(
            probability_map
        )

        statistical_output = (
            statistical_encoder(descriptors)
        )

        fusion_output = fusion(
            spatial_feature,
            statistical_output.statistical_feature,
        )

        projection_output = projector(
            fusion_output.fused_feature
        )

    global_token = projection_output.global_token

    normalized_rms = torch.sqrt(
        projection_output
        .normalized_feature
        .square()
        .mean()
    ).item()

    token_scale = float(
        projection_output.token_scale.item()
    )

    global_token_norm = float(
        global_token.norm().item()
    )

    expected_token_norm = (
        token_scale * math.sqrt(3584.0)
    )

    token_norm_error = abs(
        global_token_norm - expected_token_norm
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
        tuple(spatial_feature.shape),
    )
    print(
        "  Statistical feature:",
        tuple(
            statistical_output
            .statistical_feature
            .shape
        ),
    )
    print(
        "  Fused feature:",
        tuple(
            fusion_output.fused_feature.shape
        ),
    )
    print(
        "  Projected feature:",
        tuple(
            projection_output
            .projected_feature
            .shape
        ),
    )
    print(
        "  Global token:",
        tuple(global_token.shape),
    )

    print("\nGlobal Token diagnostics:")
    print(
        "  Token scale:",
        f"{token_scale:.8f}",
    )
    print(
        "  Normalized feature RMS:",
        f"{normalized_rms:.8f}",
    )
    print(
        "  Global token mean:",
        f"{global_token.mean().item():.8f}",
    )
    print(
        "  Global token std:",
        f"{global_token.std().item():.8f}",
    )
    print(
        "  Global token norm:",
        f"{global_token_norm:.8f}",
    )
    print(
        "  Expected token norm:",
        f"{expected_token_norm:.8f}",
    )
    print(
        "  Token norm error:",
        f"{token_norm_error:.8f}",
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
        "  Projector gradient norm:",
        f"{projector_gradient:.8f}",
    )
    print(
        "  Scale gradient:",
        f"{scale_gradient:.8f}",
    )

    checks = {
        "spatial_parameters": (
            spatial_parameters == 274_656
        ),
        "statistical_parameters": (
            statistical_parameters == 31_668
        ),
        "fusion_parameters": (
            fusion_parameters == 116_224
        ),
        "projector_parameters": (
            projector_parameters == 465_921
        ),
        "total_parameters": (
            total_parameters == 888_469
        ),
        "fused_shape": (
            fusion_output.fused_feature.shape
            == (1, 128)
        ),
        "projected_shape": (
            projection_output.projected_feature.shape
            == (1, 3584)
        ),
        "global_token_shape": (
            global_token.shape == (1, 1, 3584)
        ),
        "token_scale": (
            abs(token_scale - 0.35) < 1e-6
        ),
        "normalized_rms": (
            abs(normalized_rms - 1.0) < 1e-4
        ),
        "token_norm": (
            token_norm_error < 1e-3
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
        "projector_gradient": (
            projector_gradient > 0.0
        ),
        "scale_gradient": scale_gradient > 0.0,
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
            "source_attention_map": str(
                attention_path
            ),
            "probability_map": (
                probability_map.cpu()
            ),
            "descriptors": descriptors.cpu(),
            "spatial_feature": (
                spatial_feature.cpu()
            ),
            "statistical_feature": (
                statistical_output
                .statistical_feature
                .cpu()
            ),
            "fused_feature": (
                fusion_output
                .fused_feature
                .cpu()
            ),
            "projected_feature": (
                projection_output
                .projected_feature
                .cpu()
            ),
            "normalized_feature": (
                projection_output
                .normalized_feature
                .cpu()
            ),
            "global_token": global_token.cpu(),
            "token_scale": token_scale,
            "global_token_norm": (
                global_token_norm
            ),
            "parameter_counts": {
                "spatial_encoder": (
                    spatial_parameters
                ),
                "statistical_encoder": (
                    statistical_parameters
                ),
                "fusion": fusion_parameters,
                "projector": (
                    projector_parameters
                ),
                "total": total_parameters,
            },
        },
        output_path,
    )

    print("\nSaved untrained Global Token:")
    print(f"  {output_path}")

    if failed_checks:
        raise RuntimeError(
            "Global Token projector test failed: "
            + ", ".join(failed_checks)
        )

    print("\nGlobal Token Projector test: PASS")
    print(
        "Note: token quality is not evaluated "
        "because the Global Token Encoder is untrained."
    )


if __name__ == "__main__":
    main()