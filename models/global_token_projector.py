from dataclasses import dataclass
from math import log

import torch
import torch.nn as nn


@dataclass
class GlobalTokenProjectionOutput:
    projected_feature: torch.Tensor
    normalized_feature: torch.Tensor
    global_token: torch.Tensor
    token_scale: torch.Tensor


class GlobalTokenProjector(nn.Module):
    """
    Projects a fused 128-D global feature into one
    InternVL3 LLM token with hidden size 3584.

    Input:
        (B, 128)

    Output:
        (B, 1, 3584)
    """

    def __init__(
        self,
        input_dimension: int = 128,
        llm_hidden_dimension: int = 3584,
        initial_token_scale: float = 0.35,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()

        if initial_token_scale <= 0.0:
            raise ValueError(
                "initial_token_scale must be positive."
            )

        self.input_dimension = input_dimension
        self.llm_hidden_dimension = (
            llm_hidden_dimension
        )

        self.projection = nn.Linear(
            input_dimension,
            llm_hidden_dimension,
        )

        self.rms_norm = nn.RMSNorm(
            llm_hidden_dimension,
            eps=rms_norm_eps,
        )

        # Log parameterization guarantees a positive scale.
        self.log_token_scale = nn.Parameter(
            torch.tensor(
                log(initial_token_scale),
                dtype=torch.float32,
            )
        )

        self._initialize_projection()

    def _initialize_projection(self):
        nn.init.xavier_uniform_(
            self.projection.weight
        )

        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

    def forward(
        self,
        fused_feature: torch.Tensor,
    ) -> GlobalTokenProjectionOutput:
        if fused_feature.ndim != 2:
            raise ValueError(
                "fused_feature must have shape (B, D)."
            )

        if fused_feature.shape[1] != (
            self.input_dimension
        ):
            raise ValueError(
                f"Expected input dimension "
                f"{self.input_dimension}, but received "
                f"{fused_feature.shape[1]}."
            )

        projected_feature = self.projection(
            fused_feature
        )

        normalized_feature = self.rms_norm(
            projected_feature
        )

        token_scale = torch.exp(
            self.log_token_scale
        )

        global_token = (
            token_scale * normalized_feature
        )

        # Add the token-sequence dimension.
        global_token = global_token.unsqueeze(1)

        return GlobalTokenProjectionOutput(
            projected_feature=projected_feature,
            normalized_feature=normalized_feature,
            global_token=global_token,
            token_scale=token_scale,
        )