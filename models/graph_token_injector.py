import types

import torch


def _find_single_position(input_ids, token_id, name):
    positions = (
        input_ids[0]
        .eq(token_id)
        .nonzero(as_tuple=False)
        .flatten()
    )

    if positions.numel() != 1:
        raise ValueError(
            f"Expected one {name}, "
            f"but found {positions.numel()}."
        )

    return int(positions.item())


def _prepare_visual_input_embeddings(
    model,
    input_ids,
    pixel_values,
    visual_features=None,
):
    input_embeddings = (
        model.language_model
        .get_input_embeddings()(input_ids)
        .clone()
    )

    if visual_features is None:
        visual_features = model.extract_feature(
            pixel_values
        )

    batch_size, sequence_length, hidden_size = (
        input_embeddings.shape
    )

    flat_embeddings = input_embeddings.reshape(
        batch_size * sequence_length,
        hidden_size,
    )
    flat_input_ids = input_ids.reshape(-1)

    image_mask = (
        flat_input_ids
        == model.img_context_token_id
    )

    flattened_visual_features = (
        visual_features
        .reshape(-1, hidden_size)
        .to(
            device=flat_embeddings.device,
            dtype=flat_embeddings.dtype,
        )
    )

    if int(image_mask.sum()) != (
        flattened_visual_features.shape[0]
    ):
        raise ValueError(
            "Image placeholder count does not match "
            "the projected visual feature count."
        )

    flat_embeddings[image_mask] = (
        flattened_visual_features
    )

    return (
        flat_embeddings.reshape(
            batch_size,
            sequence_length,
            hidden_size,
        ),
        visual_features,
    )


def _reorder_paper_input(
    tokenizer,
    input_ids,
    input_embeddings,
    attention_mask,
    graph_token,
):
    if input_ids.shape[0] != 1:
        raise NotImplementedError(
            "Graph injection currently supports "
            "batch size 1 only."
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

    graph_token = graph_token.to(
        device=input_embeddings.device,
        dtype=input_embeddings.dtype,
    )

    if graph_token.ndim == 2:
        graph_token = graph_token.unsqueeze(0)

    expected_shape = (
        1,
        1,
        input_embeddings.shape[-1],
    )

    if graph_token.shape != expected_shape:
        raise ValueError(
            f"Expected Graph Token shape "
            f"{expected_shape}, "
            f"but received {tuple(graph_token.shape)}."
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
    graph_mask = torch.ones(
        (1, 1),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )

    reordered_mask = torch.cat(
        (
            prefix_mask,
            question_mask,
            graph_mask,
            image_mask,
            suffix_mask,
        ),
        dim=1,
    )

    diagnostics = {
        "original_sequence_length": (
            input_embeddings.shape[1]
        ),
        "second_pass_sequence_length": (
            reordered_embeddings.shape[1]
        ),
        "image_start": image_start,
        "image_end": image_end,
        "question_token_count": (
            user_turn_end - image_end - 1
        ),
        "graph_token_count": 1,
        "input_order": "question_graph_image",
    }

    return (
        reordered_embeddings,
        reordered_mask,
        diagnostics,
    )


@torch.inference_mode()
def chat_with_graph_token(
    model,
    tokenizer,
    pixel_values,
    question,
    graph_token,
    generation_config,
    visual_features=None,
):
    original_generate = model.generate
    captured_diagnostics = {}

    def graph_generate(
        self,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        visual_features=None,
        generation_config=None,
        output_hidden_states=None,
        **generate_kwargs,
    ):
        (
            input_embeddings,
            used_visual_features,
        ) = _prepare_visual_input_embeddings(
            model=self,
            input_ids=input_ids,
            pixel_values=pixel_values,
            visual_features=visual_features,
        )

        (
            reordered_embeddings,
            reordered_mask,
            diagnostics,
        ) = _reorder_paper_input(
            tokenizer=tokenizer,
            input_ids=input_ids,
            input_embeddings=input_embeddings,
            attention_mask=attention_mask,
            graph_token=graph_token,
        )

        diagnostics["image_context_tokens"] = int(
            input_ids.eq(
                self.img_context_token_id
            ).sum()
        )
        diagnostics["graph_token_norm"] = float(
            graph_token.float().norm()
        )
        diagnostics["image_token_norm_mean"] = float(
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
        graph_generate,
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