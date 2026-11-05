from dataclasses import dataclass

import torch


@dataclass
class GraphNodeBatch:
    visual_features: torch.Tensor
    attention_scores: torch.Tensor
    attention_weights: torch.Tensor
    normalized_positions: torch.Tensor
    normalized_boxes: torch.Tensor
    local_token_indices: torch.Tensor
    tile_indices: torch.Tensor


@torch.inference_mode()
def extract_projected_visual_features(
    model,
    pixel_values,
):
    visual_features = model.extract_feature(
        pixel_values
    )

    if visual_features.ndim != 3:
        raise ValueError(
            "Expected visual features with shape "
            "[image_inputs, tokens_per_tile, hidden_size], "
            f"but received {tuple(visual_features.shape)}."
        )

    if not torch.isfinite(visual_features).all():
        raise ValueError(
            "Visual features contain NaN or Inf."
        )

    return visual_features


def build_graph_nodes(
    projected_visual_features,
    selection,
    num_local_tiles,
):
    image_inputs, tokens_per_tile, hidden_size = (
        projected_visual_features.shape
    )

    expected_tokens_per_tile = (
        selection.token_grid[0]
        * selection.token_grid[1]
    )

    if tokens_per_tile != expected_tokens_per_tile:
        raise ValueError(
            f"Expected {expected_tokens_per_tile} tokens "
            f"per tile, but received {tokens_per_tile}."
        )

    if num_local_tiles > image_inputs:
        raise ValueError(
            f"num_local_tiles={num_local_tiles} exceeds "
            f"image_inputs={image_inputs}."
        )

    local_visual_features = (
        projected_visual_features[:num_local_tiles]
        .reshape(-1, hidden_size)
    )

    expected_local_tokens = (
        num_local_tiles * tokens_per_tile
    )

    if local_visual_features.shape[0] != expected_local_tokens:
        raise RuntimeError(
            "Unexpected flattened local feature count."
        )

    device = local_visual_features.device

    local_token_indices = torch.tensor(
        [
            region.local_token_index
            for region in selection.regions
        ],
        dtype=torch.long,
        device=device,
    )

    if int(local_token_indices.max()) >= expected_local_tokens:
        raise IndexError(
            "A selected local token index is out of range."
        )

    node_visual_features = (
        local_visual_features.index_select(
            dim=0,
            index=local_token_indices,
        )
    )

    attention_scores = torch.tensor(
        [
            region.attention_score
            for region in selection.regions
        ],
        dtype=torch.float32,
        device=device,
    )

    attention_weights = torch.tensor(
        [
            region.selected_weight
            for region in selection.regions
        ],
        dtype=torch.float32,
        device=device,
    )

    normalized_positions = torch.tensor(
        [
            region.normalized_center_xy
            for region in selection.regions
        ],
        dtype=torch.float32,
        device=device,
    )

    image_width, image_height = selection.image_size

    normalized_boxes = torch.tensor(
        [
            (
                region.bbox_xyxy[0] / image_width,
                region.bbox_xyxy[1] / image_height,
                region.bbox_xyxy[2] / image_width,
                region.bbox_xyxy[3] / image_height,
            )
            for region in selection.regions
        ],
        dtype=torch.float32,
        device=device,
    )

    tile_indices = torch.tensor(
        [
            region.tile_index
            for region in selection.regions
        ],
        dtype=torch.long,
        device=device,
    )

    if not torch.isclose(
        attention_weights.sum(),
        torch.tensor(
            1.0,
            device=device,
            dtype=torch.float32,
        ),
        atol=1e-6,
    ):
        raise RuntimeError(
            "Selected attention weights do not sum to 1."
        )

    if not torch.isfinite(
        node_visual_features
    ).all():
        raise RuntimeError(
            "Graph node features contain NaN or Inf."
        )

    return GraphNodeBatch(
        visual_features=node_visual_features,
        attention_scores=attention_scores,
        attention_weights=attention_weights,
        normalized_positions=normalized_positions,
        normalized_boxes=normalized_boxes,
        local_token_indices=local_token_indices,
        tile_indices=tile_indices,
    )