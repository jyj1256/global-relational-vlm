from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class WeightedGraph:
    node_features: torch.Tensor
    attention_profiles: torch.Tensor
    spatial_similarity: torch.Tensor
    visual_similarity: torch.Tensor
    attention_similarity: torch.Tensor
    dense_edge_weights: torch.Tensor
    retained_edge_mask: torch.Tensor
    pruned_edge_weights: torch.Tensor
    normalized_edge_weights: torch.Tensor


def _rescaled_cosine_similarity(features):
    normalized = F.normalize(
        features.float(),
        p=2,
        dim=-1,
        eps=1e-12,
    )
    cosine = normalized @ normalized.T

    return ((cosine + 1.0) / 2.0).clamp(
        min=0.0,
        max=1.0,
    )


def build_weighted_graph(
    visual_features,
    normalized_positions,
    attention_scores,
    attention_profiles,
    sigma_p=0.25,
    threshold=0.70,
    spatial_weight=0.35,
    visual_weight=0.20,
    attention_weight=0.45,
):
    visual_features = visual_features.float()
    normalized_positions = normalized_positions.float()
    attention_scores = attention_scores.float()
    attention_profiles = attention_profiles.float()

    node_count = visual_features.shape[0]

    if normalized_positions.shape != (node_count, 2):
        raise ValueError(
            "normalized_positions must have shape [K, 2]."
        )

    if attention_scores.shape != (node_count,):
        raise ValueError(
            "attention_scores must have shape [K]."
        )

    if attention_profiles.shape[0] != node_count:
        raise ValueError(
            "Attention profile count does not match K."
        )

    if sigma_p <= 0.0:
        raise ValueError("sigma_p must be positive.")

    coefficient_sum = (
        spatial_weight
        + visual_weight
        + attention_weight
    )

    if abs(coefficient_sum - 1.0) > 1e-6:
        raise ValueError(
            "Edge coefficients must sum to 1."
        )

    if not all(
        torch.isfinite(tensor).all()
        for tensor in (
            visual_features,
            normalized_positions,
            attention_scores,
            attention_profiles,
        )
    ):
        raise ValueError(
            "Graph inputs contain NaN or Inf."
        )

    position_difference = (
        normalized_positions[:, None, :]
        - normalized_positions[None, :, :]
    )
    squared_distance = (
        position_difference
        .pow(2)
        .sum(dim=-1)
    )

    spatial_similarity = torch.exp(
        -squared_distance
        / (2.0 * sigma_p * sigma_p)
    )

    visual_similarity = (
        _rescaled_cosine_similarity(
            visual_features
        )
    )
    attention_similarity = (
        _rescaled_cosine_similarity(
            attention_profiles
        )
    )

    diagonal_mask = torch.eye(
        node_count,
        dtype=torch.bool,
        device=visual_features.device,
    )

    for matrix in (
        spatial_similarity,
        visual_similarity,
        attention_similarity,
    ):
        matrix.masked_fill_(diagonal_mask, 0.0)

    dense_edge_weights = (
        spatial_weight * spatial_similarity
        + visual_weight * visual_similarity
        + attention_weight * attention_similarity
    )
    dense_edge_weights.masked_fill_(
        diagonal_mask,
        0.0,
    )

    threshold_mask = (
        dense_edge_weights >= threshold
    )
    threshold_mask.masked_fill_(
        diagonal_mask,
        False,
    )

    retained_edge_mask = threshold_mask.clone()

    strongest_candidates = (
        dense_edge_weights.clone()
    )
    strongest_candidates.masked_fill_(
        diagonal_mask,
        float("-inf"),
    )

    strongest_neighbors = (
        strongest_candidates.argmax(dim=1)
    )

    for node_index in range(node_count):
        neighbor_index = int(
            strongest_neighbors[node_index]
        )

        retained_edge_mask[
            node_index,
            neighbor_index,
        ] = True
        retained_edge_mask[
            neighbor_index,
            node_index,
        ] = True

    retained_edge_mask = (
        retained_edge_mask
        | retained_edge_mask.T
    )

    pruned_edge_weights = (
        dense_edge_weights
        * retained_edge_mask.float()
    )

    row_sums = pruned_edge_weights.sum(
        dim=1,
        keepdim=True,
    )

    if torch.any(row_sums <= 0.0):
        raise RuntimeError(
            "An isolated node remains after fallback."
        )

    normalized_edge_weights = (
        pruned_edge_weights / row_sums
    )

    node_features = torch.cat(
        (
            visual_features,
            normalized_positions,
            attention_scores.unsqueeze(-1),
        ),
        dim=-1,
    )

    return WeightedGraph(
        node_features=node_features,
        attention_profiles=attention_profiles,
        spatial_similarity=spatial_similarity,
        visual_similarity=visual_similarity,
        attention_similarity=attention_similarity,
        dense_edge_weights=dense_edge_weights,
        retained_edge_mask=retained_edge_mask,
        pruned_edge_weights=pruned_edge_weights,
        normalized_edge_weights=(
            normalized_edge_weights
        ),
    )