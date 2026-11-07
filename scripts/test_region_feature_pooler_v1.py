import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from vlm.models.object_graph_v1.region_feature_pooler import (
    RegionFeaturePooler,
)


def close(value, expected, tolerance=1e-6):
    return abs(
        float(value) - float(expected)
    ) <= tolerance


def expect_error(function):
    try:
        function()
    except (ValueError, RuntimeError):
        return True

    return False


def main():
    torch.manual_seed(2026)

    token_grid_size = 16
    tile_grid = (2, 1)

    local_tile_count = (
        tile_grid[0] * tile_grid[1]
    )
    tokens_per_tile = (
        token_grid_size ** 2
    )
    local_token_count = (
        local_tile_count
        * tokens_per_tile
    )

    visual_dimension = 8
    profile_shape = (2, 3)

    token_values = torch.arange(
        local_token_count,
        dtype=torch.float32,
    )

    local_visual = torch.stack(
        [
            token_values,
            torch.ones_like(token_values),
            token_values / 512.0,
            token_values.square() / 512.0,
            torch.sin(token_values / 10.0),
            torch.cos(token_values / 10.0),
            token_values % 16,
            token_values // 16,
        ],
        dim=-1,
    )

    thumbnail_visual = torch.full(
        (
            1,
            tokens_per_tile,
            visual_dimension,
        ),
        1_000_000.0,
        dtype=torch.float32,
    )

    projected_visual_features = (
        torch.cat(
            [
                local_visual.reshape(
                    local_tile_count,
                    tokens_per_tile,
                    visual_dimension,
                ),
                thumbnail_visual,
            ],
            dim=0,
        )
        .requires_grad_(True)
    )

    attention_profiles = (
        torch.arange(
            local_token_count
            * profile_shape[0]
            * profile_shape[1],
            dtype=torch.float32,
        )
        .reshape(
            local_token_count,
            *profile_shape,
        )
        .div(1000.0)
        .requires_grad_(True)
    )

    local_attention = torch.zeros(
        local_token_count,
        dtype=torch.float32,
    )

    local_attention[17] = 0.7
    local_attention[256] = 0.2
    local_attention[511] = 0.1

    mosaic_width = (
        tile_grid[0] * token_grid_size
    )
    mosaic_height = (
        tile_grid[1] * token_grid_size
    )

    boxes = [
        (
            0.0,
            0.0,
            2.0 / mosaic_width,
            2.0 / mosaic_height,
        ),
        (
            0.5,
            0.0,
            0.5 + 1.0 / mosaic_width,
            1.0 / mosaic_height,
        ),
        (
            0.0,
            0.0,
            0.5,
            1.0,
        ),
        (
            0.0,
            0.0,
            1.0,
            1.0,
        ),
        (
            0.0,
            0.0,
            0.5 / mosaic_width,
            1.0 / mosaic_height,
        ),
    ]

    pooler = RegionFeaturePooler(
        token_grid_size=token_grid_size,
        eps=1e-12,
    )

    output = pooler(
        normalized_boxes=boxes,
        local_attention=local_attention,
        projected_visual_features=(
            projected_visual_features
        ),
        attention_profiles=(
            attention_profiles
        ),
        tile_grid=tile_grid,
    )

    (
        two_by_two,
        second_tile_patch,
        first_tile,
        full_image,
        half_patch,
    ) = output.region_features

    expected_two_by_two_indices = (
        0,
        1,
        16,
        17,
    )

    expected_two_by_two_mean = (
        0.0 + 1.0 + 16.0 + 17.0
    ) / 4.0

    expected_two_by_two_profile = (
        attention_profiles.detach()[
            list(expected_two_by_two_indices)
        ]
        .mean(dim=0)
    )

    expected_full_attention_feature = (
        17.0 * 0.7
        + 256.0 * 0.2
        + 511.0 * 0.1
    )

    patch_area = 1.0 / (
        mosaic_width * mosaic_height
    )

    all_tensors = []

    for region in output.region_features:
        all_tensors.extend(
            [
                region.intersection_areas,
                region.geometric_weights,
                region.attention_weights,
                region.geometric_visual_feature,
                region.attention_visual_feature,
                region.pooled_attention_profile,
            ]
        )

    finite_outputs = all(
        bool(torch.isfinite(tensor).all())
        for tensor in all_tensors
    )

    weights_normalized = all(
        close(
            region.geometric_weights.sum(),
            1.0,
        )
        and close(
            region.attention_weights.sum(),
            1.0,
        )
        for region in output.region_features
    )

    invalid_box_rejected = expect_error(
        lambda: pooler(
            normalized_boxes=[
                (-0.1, 0.0, 0.2, 0.2)
            ],
            local_attention=local_attention,
            projected_visual_features=(
                projected_visual_features
            ),
            attention_profiles=(
                attention_profiles
            ),
            tile_grid=tile_grid,
        )
    )

    profile_mismatch_rejected = expect_error(
        lambda: pooler(
            normalized_boxes=[
                (0.0, 0.0, 1.0, 1.0)
            ],
            local_attention=local_attention,
            projected_visual_features=(
                projected_visual_features
            ),
            attention_profiles=(
                attention_profiles[:-1]
            ),
            tile_grid=tile_grid,
        )
    )

    loss = torch.zeros(
        (),
        dtype=torch.float32,
    )

    for region in output.region_features:
        loss = (
            loss
            + region.geometric_visual_feature
            .square()
            .mean()
            + region.attention_visual_feature
            .square()
            .mean()
            + region.pooled_attention_profile
            .square()
            .mean()
        )

    loss.backward()

    local_visual_gradient = (
        projected_visual_features.grad[
            :local_tile_count
        ]
    )

    thumbnail_gradient = (
        projected_visual_features.grad[
            local_tile_count:
        ]
    )

    checks = {
        "region_count": (
            len(output.region_features) == 5
        ),
        "tile_grid": (
            output.tile_grid == tile_grid
        ),
        "local_token_count": (
            output.local_token_count
            == local_token_count
        ),
        "visual_dimension": (
            output.visual_dimension
            == visual_dimension
        ),
        "profile_shape": (
            output.attention_profile_shape
            == profile_shape
        ),
        "two_by_two_indices": (
            two_by_two.local_token_indices
            == expected_two_by_two_indices
        ),
        "two_by_two_geometric_mean": close(
            two_by_two
            .geometric_visual_feature[0],
            expected_two_by_two_mean,
        ),
        "two_by_two_attention_pool": close(
            two_by_two
            .attention_visual_feature[0],
            17.0,
        ),
        "two_by_two_profile": bool(
            torch.allclose(
                two_by_two
                .pooled_attention_profile,
                expected_two_by_two_profile,
                atol=1e-6,
                rtol=0.0,
            )
        ),
        "second_tile_mapping": (
            second_tile_patch
            .local_token_indices
            == (256,)
        ),
        "second_tile_feature": close(
            second_tile_patch
            .geometric_visual_feature[0],
            256.0,
        ),
        "first_tile_count": (
            len(
                first_tile
                .local_token_indices
            )
            == 256
        ),
        "first_tile_attention_mass": close(
            first_tile.attention_mass,
            0.7,
        ),
        "full_image_count": (
            len(
                full_image
                .local_token_indices
            )
            == 512
        ),
        "full_image_geometric_mean": close(
            full_image
            .geometric_visual_feature[0],
            255.5,
        ),
        "full_image_attention_pool": close(
            full_image
            .attention_visual_feature[0],
            expected_full_attention_feature,
            tolerance=1e-5,
        ),
        "full_image_attention_mass": close(
            full_image.attention_mass,
            1.0,
        ),
        "full_image_coverage": close(
            full_image.coverage_area,
            1.0,
        ),
        "half_patch_index": (
            half_patch.local_token_indices
            == (0,)
        ),
        "half_patch_area": close(
            half_patch.coverage_area,
            patch_area * 0.5,
        ),
        "weights_normalized": (
            weights_normalized
        ),
        "finite_outputs": finite_outputs,
        "invalid_box_rejected": (
            invalid_box_rejected
        ),
        "profile_mismatch_rejected": (
            profile_mismatch_rejected
        ),
        "visual_gradient": bool(
            local_visual_gradient is not None
            and torch.isfinite(
                local_visual_gradient
            ).all()
            and local_visual_gradient
            .abs()
            .sum()
            .item() > 0.0
        ),
        "profile_gradient": bool(
            attention_profiles.grad is not None
            and torch.isfinite(
                attention_profiles.grad
            ).all()
            and attention_profiles.grad
            .abs()
            .sum()
            .item() > 0.0
        ),
        "thumbnail_excluded": bool(
            thumbnail_gradient is not None
            and thumbnail_gradient
            .abs()
            .max()
            .item() == 0.0
        ),
    }

    print(
        "Region Feature Pooler summary:"
    )
    print(
        "  Tile grid:",
        output.tile_grid,
    )
    print(
        "  Local tokens:",
        output.local_token_count,
    )
    print(
        "  Input visual tiles:",
        tuple(
            projected_visual_features.shape
        ),
    )
    print(
        "  Visual dimension:",
        output.visual_dimension,
    )
    print(
        "  Attention profile shape:",
        output.attention_profile_shape,
    )
    print(
        "  Regions:",
        len(output.region_features),
    )

    print("\nRegion diagnostics:")

    for region in output.region_features:
        print(
            f"  R{region.region_index}: "
            f"patches="
            f"{len(region.local_token_indices)}, "
            f"coverage={region.coverage_area:.8f}, "
            f"attention_mass="
            f"{region.attention_mass:.8f}, "
            f"density="
            f"{region.attention_density:.8f}, "
            f"peak="
            f"{region.attention_peak:.8f}"
        )

    print("\nExact mapping diagnostics:")
    print(
        "  Two-by-two indices:",
        two_by_two.local_token_indices,
    )
    print(
        "  Two-by-two geometric mean:",
        float(
            two_by_two
            .geometric_visual_feature[0]
        ),
    )
    print(
        "  Two-by-two attention pool:",
        float(
            two_by_two
            .attention_visual_feature[0]
        ),
    )
    print(
        "  Second-tile indices:",
        second_tile_patch
        .local_token_indices,
    )
    print(
        "  Full-image geometric mean:",
        float(
            full_image
            .geometric_visual_feature[0]
        ),
    )
    print(
        "  Full-image attention pool:",
        float(
            full_image
            .attention_visual_feature[0]
        ),
    )

    print("\nGradient diagnostics:")
    print(
        "  Test loss:",
        f"{loss.item():.8f}",
    )
    print(
        "  Local visual gradient:",
        f"{local_visual_gradient.norm().item():.8f}",
    )
    print(
        "  Thumbnail gradient:",
        f"{thumbnail_gradient.norm().item():.8f}",
    )
    print(
        "  Attention profile gradient:",
        f"{attention_profiles.grad.norm().item():.8f}",
    )

    print("\nChecks:")

    for name, passed in checks.items():
        print(
            f"  {name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    if not all(checks.values()):
        failed = [
            name
            for name, passed
            in checks.items()
            if not passed
        ]

        raise RuntimeError(
            "Region Feature Pooler test "
            f"failed: {failed}"
        )

    print(
        "\nRegion Feature Pooler test: PASS"
    )
    print(
        "Note: this validates coordinate "
        "mapping and pooling, not region quality."
    )


if __name__ == "__main__":
    main()