import types

import torch

from vlm.models.graph_token_injector import (
    _find_single_position,
    _prepare_visual_input_embeddings,
)


def _prepare_auxiliary_token(
    token,
    token_name,
    input_embeddings,
):
    token = torch.as_tensor(token)

    if token.ndim == 1:
        token = token.unsqueeze(0).unsqueeze(0)

    elif token.ndim == 2:
        token = token.unsqueeze(0)

    expected_shape = (
        1,
        1,
        input_embeddings.shape[-1],
    )

    if tuple(token.shape) != expected_shape:
        raise ValueError(
            f"Expected {token_name} shape "
            f"{expected_shape}, but received "
            f"{tuple(token.shape)}."
        )

    return token.to(
        device=input_embeddings.device,
        dtype=input_embeddings.dtype,
    )


def _reorder_global_graph_input(
    tokenizer,
    input_ids,
    input_embeddings,
    attention_mask,
    global_token,
    graph_token,
):
    """
    Construct the second-pass order:

        Question -> Global -> Graph -> Image

    Prompt prefix and suffix tokens are preserved.
    """

    if input_ids.shape[0] != 1:
        raise NotImplementedError(
            "Global/Graph injection currently "
            "supports batch size 1 only."
        )

    image_start_id = (
        tokenizer.convert_tokens_to_ids("<img>")
    )

    image_end_id = (
        tokenizer.convert_tokens_to_ids("</img>")
    )

    turn_end_id = (
        tokenizer.convert_tokens_to_ids(
            "<|im_end|>"
        )
    )

    image_start = _find_single_position(
        input_ids,
        image_start_id,
        "<img>",
    )

    image_end = _find_single_position(
        input_ids,
        image_end_id,
        "</img>",
    )

    turn_end_positions = (
        input_ids[0]
        .eq(turn_end_id)
        .nonzero(as_tuple=False)
        .flatten()
    )

    valid_turn_ends = turn_end_positions[
        turn_end_positions > image_end
    ]

    if valid_turn_ends.numel() == 0:
        raise ValueError(
            "Could not locate the user turn end."
        )

    user_turn_end = int(
        valid_turn_ends[-1].item()
    )

    global_token = _prepare_auxiliary_token(
        token=global_token,
        token_name="Global Token",
        input_embeddings=input_embeddings,
    )

    graph_token = _prepare_auxiliary_token(
        token=graph_token,
        token_name="Graph Token",
        input_embeddings=input_embeddings,
    )

    prefix_embeddings = input_embeddings[
        :,
        :image_start,
        :,
    ]

    image_embeddings = input_embeddings[
        :,
        image_start:image_end + 1,
        :,
    ]

    question_embeddings = input_embeddings[
        :,
        image_end + 1:user_turn_end,
        :,
    ]

    suffix_embeddings = input_embeddings[
        :,
        user_turn_end:,
        :,
    ]

    reordered_embeddings = torch.cat(
        (
            prefix_embeddings,
            question_embeddings,
            global_token,
            graph_token,
            image_embeddings,
            suffix_embeddings,
        ),
        dim=1,
    )

    prefix_mask = attention_mask[
        :,
        :image_start,
    ]

    image_mask = attention_mask[
        :,
        image_start:image_end + 1,
    ]

    question_mask = attention_mask[
        :,
        image_end + 1:user_turn_end,
    ]

    suffix_mask = attention_mask[
        :,
        user_turn_end:,
    ]

    global_mask = torch.ones(
        (1, 1),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )

    graph_mask = torch.ones(
        (1, 1),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )

    reordered_mask = torch.cat(
        (
            prefix_mask,
            question_mask,
            global_mask,
            graph_mask,
            image_mask,
            suffix_mask,
        ),
        dim=1,
    )

    question_token_count = (
        user_turn_end - image_end - 1
    )

    auxiliary_start = (
        prefix_embeddings.shape[1]
        + question_embeddings.shape[1]
    )

    diagnostics = {
        "original_sequence_length": int(
            input_embeddings.shape[1]
        ),
        "second_pass_sequence_length": int(
            reordered_embeddings.shape[1]
        ),
        "image_start": image_start,
        "image_end": image_end,
        "question_token_count": (
            question_token_count
        ),
        "global_token_count": 1,
        "graph_token_count": 1,
        "global_token_index": auxiliary_start,
        "graph_token_index": (
            auxiliary_start + 1
        ),
        "reordered_image_start": (
            auxiliary_start + 2
        ),
        "input_order": (
            "question_global_graph_image"
        ),
    }

    return (
        reordered_embeddings,
        reordered_mask,
        diagnostics,
    )


@torch.inference_mode()
def chat_with_global_graph_tokens(
    model,
    tokenizer,
    pixel_values,
    question,
    global_token,
    graph_token,
    generation_config,
    visual_features=None,
):
    """
    Run the second InternVL pass with one Global
    Token and one Graph Token.
    """

    original_generate = model.generate
    captured_diagnostics = {}

    precomputed_visual_features = visual_features

    def global_graph_generate(
        self,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        visual_features=None,
        generation_config=None,
        output_hidden_states=None,
        **generate_kwargs,
    ):
        effective_visual_features = (
            visual_features
        )

        if effective_visual_features is None:
            effective_visual_features = (
                precomputed_visual_features
            )

        (
            input_embeddings,
            used_visual_features,
        ) = _prepare_visual_input_embeddings(
            model=self,
            input_ids=input_ids,
            pixel_values=pixel_values,
            visual_features=(
                effective_visual_features
            ),
        )

        (
            reordered_embeddings,
            reordered_mask,
            diagnostics,
        ) = _reorder_global_graph_input(
            tokenizer=tokenizer,
            input_ids=input_ids,
            input_embeddings=input_embeddings,
            attention_mask=attention_mask,
            global_token=global_token,
            graph_token=graph_token,
        )

        diagnostics[
            "image_context_tokens"
        ] = int(
            input_ids.eq(
                self.img_context_token_id
            ).sum()
        )

        diagnostics["global_token_norm"] = float(
            global_token.float().norm()
        )

        diagnostics["graph_token_norm"] = float(
            graph_token.float().norm()
        )

        diagnostics[
            "image_token_norm_mean"
        ] = float(
            used_visual_features
            .float()
            .norm(dim=-1)
            .mean()
        )

        captured_diagnostics.update(
            diagnostics
        )

        if output_hidden_states is not None:
            generate_kwargs[
                "output_hidden_states"
            ] = output_hidden_states

        generate_kwargs.setdefault(
            "use_cache",
            True,
        )

        return self.language_model.generate(
            inputs_embeds=reordered_embeddings,
            attention_mask=reordered_mask,
            generation_config=generation_config,
            **generate_kwargs,
        )

    model.generate = types.MethodType(
        global_graph_generate,
        model,
    )

    try:
        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=f"<image>\n{question}",
            generation_config=(
                generation_config
            ),
        )

    finally:
        model.generate = original_generate

    return response, captured_diagnostics