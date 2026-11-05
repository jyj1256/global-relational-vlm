import torch


def _to_int_list(values):
    if torch.is_tensor(values):
        values = (
            values.detach()
            .cpu()
            .reshape(-1)
            .tolist()
        )

    return [int(value) for value in values]


def extract_local_attention_profiles(
    attention_result,
):
    per_layer_head = attention_result.per_layer_head

    if per_layer_head.ndim != 3:
        raise ValueError(
            "per_layer_head must have shape "
            "[layers, heads, image_tokens]."
        )

    image_positions = _to_int_list(
        attention_result.image_positions
    )
    local_positions = _to_int_list(
        attention_result.local_image_positions
    )

    if per_layer_head.shape[-1] != len(image_positions):
        raise ValueError(
            "Image-position count does not match "
            "the attention tensor."
        )

    if len(set(image_positions)) != len(image_positions):
        raise ValueError(
            "Duplicate positions exist in image_positions."
        )

    position_to_index = {
        position: index
        for index, position in enumerate(image_positions)
    }

    missing_positions = [
        position
        for position in local_positions
        if position not in position_to_index
    ]

    if missing_positions:
        raise ValueError(
            "Local positions missing from image positions: "
            f"{missing_positions[:10]}"
        )

    local_indices = torch.tensor(
        [
            position_to_index[position]
            for position in local_positions
        ],
        dtype=torch.long,
        device=per_layer_head.device,
    )

    local_profiles = (
        per_layer_head
        .index_select(
            dim=-1,
            index=local_indices,
        )
        .permute(2, 0, 1)
        .contiguous()
        .float()
    )

    if not torch.isfinite(local_profiles).all():
        raise ValueError(
            "Local attention profiles contain NaN or Inf."
        )

    return local_profiles