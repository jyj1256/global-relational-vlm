import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


VLM_ROOT = Path(__file__).resolve().parents[1]

if str(VLM_ROOT) not in sys.path:
    sys.path.insert(0, str(VLM_ROOT))


from models.attention_descriptors import (
    AttentionDescriptorExtractor,
)
from models.spatial_attention_encoder import (
    SpatialAttentionEncoder,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test the lightweight spatial "
            "attention CNN."
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
            "untrained_spatial_feature_test.pt"
        ),
    )

    return parser.parse_args()


def calculate_gradient_norm(
    model: torch.nn.Module,
) -> float:
    squared_norm = 0.0

    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        gradient = parameter.grad.detach().float()
        squared_norm += float(
            gradient.square().sum().item()
        )

    return squared_norm ** 0.5


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
        )
    )

    with torch.no_grad():
        descriptor_result = descriptor_extractor(
            torch.from_numpy(attention_array)
        )

    probability_map = (
        descriptor_result.probability_map
    )

    model = SpatialAttentionEncoder(
        map_size=32,
        output_dimension=128,
        dropout=0.1,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    model.train()

    spatial_feature, feature_map = model(
        probability_map,
        return_feature_map=True,
    )

    test_loss = spatial_feature.square().mean()
    test_loss.backward()

    gradient_norm = calculate_gradient_norm(model)

    model.eval()

    with torch.no_grad():
        original_feature = model(probability_map)

        flipped_probability_map = torch.flip(
            probability_map,
            dims=(-1,),
        )

        flipped_feature = model(
            flipped_probability_map
        )

        scaled_input = model.prepare_input(
            probability_map
        )

    position_difference = torch.linalg.vector_norm(
        original_feature - flipped_feature
    ).item()

    position_cosine = F.cosine_similarity(
        original_feature,
        flipped_feature,
        dim=-1,
    ).item()

    has_nan = bool(
        torch.isnan(original_feature).any().item()
    )

    has_inf = bool(
        torch.isinf(original_feature).any().item()
    )

    print("Spatial Attention Encoder summary:")
    print(
        "  Probability map:",
        tuple(probability_map.shape),
    )
    print(
        "  Scaled input:",
        tuple(scaled_input.shape),
    )
    print(
        "  Scaled input mean:",
        f"{scaled_input.mean().item():.8f}",
    )
    print(
        "  CNN feature map:",
        tuple(feature_map.shape),
    )
    print(
        "  Spatial feature:",
        tuple(original_feature.shape),
    )
    print(
        "  Trainable parameters:",
        f"{parameter_count:,}",
    )

    print("\nOutput diagnostics:")
    print(
        "  Feature mean:",
        f"{original_feature.mean().item():.8f}",
    )
    print(
        "  Feature std:",
        f"{original_feature.std().item():.8f}",
    )
    print(
        "  Feature norm:",
        f"{original_feature.norm().item():.8f}",
    )
    print(
        "  Test loss:",
        f"{test_loss.item():.8f}",
    )
    print(
        "  Gradient norm:",
        f"{gradient_norm:.8f}",
    )
    print("  NaN:", has_nan)
    print("  Inf:", has_inf)

    print("\nPosition sensitivity:")
    print(
        "  Original vs horizontal flip L2:",
        f"{position_difference:.8f}",
    )
    print(
        "  Original vs horizontal flip cosine:",
        f"{position_cosine:.8f}",
    )

    checks = {
        "scaled_input_mean": (
            abs(scaled_input.mean().item() - 1.0)
            < 1e-5
        ),
        "feature_map_shape": (
            feature_map.shape == (1, 128, 4, 4)
        ),
        "output_shape": (
            original_feature.shape == (1, 128)
        ),
        "parameter_count": (
            parameter_count == 274_656
        ),
        "finite": not has_nan and not has_inf,
        "gradient": gradient_norm > 0.0,
        "position_sensitive": (
            position_difference > 1e-6
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
            "source_attention_map": str(
                attention_path
            ),
            "probability_map": (
                probability_map.cpu()
            ),
            "scaled_input": scaled_input.cpu(),
            "cnn_feature_map": feature_map.detach().cpu(),
            "spatial_feature": (
                original_feature.cpu()
            ),
            "parameter_count": parameter_count,
            "position_difference": (
                position_difference
            ),
            "position_cosine": position_cosine,
        },
        output_path,
    )

    print("\nSaved untrained test artifact:")
    print(f"  {output_path}")

    if failed_checks:
        raise RuntimeError(
            "Spatial encoder test failed: "
            + ", ".join(failed_checks)
        )

    print("\nSpatial Attention Encoder test: PASS")
    print(
        "Note: feature quality is not evaluated "
        "because the encoder is untrained."
    )


if __name__ == "__main__":
    main()