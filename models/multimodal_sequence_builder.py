from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class TrainingSequenceBatch:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    diagnostics: Dict


def _find_single_position(
    input_ids,
    token_id,
    token_name,
):
    positions = (
        input_ids[0]
        .eq(token_id)
        .nonzero(as_tuple=False)
        .flatten()
    )

    if positions.numel() != 1:
        raise ValueError(
            f"Expected exactly one {token_name}, "
            f"but found {positions.numel()}."
        )

    return int(positions.item())


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


def build_training_sequence(
    tokenizer,
    prompt_input_ids,
    prompt_input_embeddings,
    prompt_attention_mask,
    global_token,
    graph_token,
    answer_input_ids,
    answer_embeddings,
    answer_attention_mask=None,
):
    """
    Build the teacher-forcing sequence:

        Prefix
        Question
        Global Token
        Graph Token
        Image Tokens
        Assistant Prefix
        Answer Tokens

    Only Answer Tokens receive valid labels.
    """

    if prompt_input_ids.ndim != 2:
        raise ValueError(
            "prompt_input_ids must have shape (B, L)."
        )

    if prompt_input_ids.shape[0] != 1:
        raise NotImplementedError(
            "Training sequence currently supports "
            "batch size 1 only."
        )

    if prompt_input_embeddings.ndim != 3:
        raise ValueError(
            "prompt_input_embeddings must have "
            "shape (B, L, D)."
        )

    if prompt_attention_mask.ndim != 2:
        raise ValueError(
            "prompt_attention_mask must have "
            "shape (B, L)."
        )

    if answer_input_ids.ndim != 2:
        raise ValueError(
            "answer_input_ids must have "
            "shape (B, A)."
        )

    if answer_embeddings.ndim != 3:
        raise ValueError(
            "answer_embeddings must have "
            "shape (B, A, D)."
        )

    prompt_length = prompt_input_ids.shape[1]

    if prompt_input_embeddings.shape[:2] != (
        1,
        prompt_length,
    ):
        raise ValueError(
            "Prompt ID and embedding shapes do not match."
        )

    if prompt_attention_mask.shape != (
        1,
        prompt_length,
    ):
        raise ValueError(
            "Prompt ID and mask shapes do not match."
        )

    answer_length = answer_input_ids.shape[1]

    if answer_embeddings.shape[:2] != (
        1,
        answer_length,
    ):
        raise ValueError(
            "Answer ID and embedding shapes do not match."
        )

    if answer_embeddings.shape[-1] != (
        prompt_input_embeddings.shape[-1]
    ):
        raise ValueError(
            "Prompt and answer embedding dimensions "
            "do not match."
        )

    device = prompt_input_embeddings.device

    prompt_input_ids = prompt_input_ids.to(
        device=device
    )

    prompt_attention_mask = (
        prompt_attention_mask.to(device=device)
    )

    answer_input_ids = answer_input_ids.to(
        device=device,
        dtype=torch.long,
    )

    answer_embeddings = answer_embeddings.to(
        device=device,
        dtype=prompt_input_embeddings.dtype,
    )

    if answer_attention_mask is None:
        answer_attention_mask = torch.ones(
            (1, answer_length),
            dtype=prompt_attention_mask.dtype,
            device=device,
        )
    else:
        answer_attention_mask = (
            answer_attention_mask.to(
                device=device,
                dtype=prompt_attention_mask.dtype,
            )
        )

    if answer_attention_mask.shape != (
        1,
        answer_length,
    ):
        raise ValueError(
            "Unexpected answer_attention_mask shape."
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
        prompt_input_ids,
        image_start_id,
        "<img>",
    )

    image_end = _find_single_position(
        prompt_input_ids,
        image_end_id,
        "</img>",
    )

    turn_end_positions = (
        prompt_input_ids[0]
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
        global_token,
        "Global Token",
        prompt_input_embeddings,
    )

    graph_token = _prepare_auxiliary_token(
        graph_token,
        "Graph Token",
        prompt_input_embeddings,
    )

    prefix_embeddings = prompt_input_embeddings[
        :,
        :image_start,
        :,
    ]

    image_embeddings = prompt_input_embeddings[
        :,
        image_start:image_end + 1,
        :,
    ]

    question_embeddings = (
        prompt_input_embeddings[
            :,
            image_end + 1:user_turn_end,
            :,
        ]
    )

    suffix_embeddings = prompt_input_embeddings[
        :,
        user_turn_end:,
        :,
    ]

    reordered_prompt_embeddings = torch.cat(
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

    prefix_mask = prompt_attention_mask[
        :,
        :image_start,
    ]

    image_mask = prompt_attention_mask[
        :,
        image_start:image_end + 1,
    ]

    question_mask = prompt_attention_mask[
        :,
        image_end + 1:user_turn_end,
    ]

    suffix_mask = prompt_attention_mask[
        :,
        user_turn_end:,
    ]

    global_mask = torch.ones(
        (1, 1),
        dtype=prompt_attention_mask.dtype,
        device=device,
    )

    graph_mask = torch.ones(
        (1, 1),
        dtype=prompt_attention_mask.dtype,
        device=device,
    )

    reordered_prompt_mask = torch.cat(
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

    inputs_embeds = torch.cat(
        (
            reordered_prompt_embeddings,
            answer_embeddings,
        ),
        dim=1,
    )

    attention_mask = torch.cat(
        (
            reordered_prompt_mask,
            answer_attention_mask,
        ),
        dim=1,
    )

    prompt_label_mask = torch.full(
        (
            1,
            reordered_prompt_embeddings.shape[1],
        ),
        fill_value=-100,
        dtype=torch.long,
        device=device,
    )

    answer_labels = answer_input_ids.clone()

    answer_labels = answer_labels.masked_fill(
        answer_attention_mask.eq(0),
        -100,
    )

    labels = torch.cat(
        (
            prompt_label_mask,
            answer_labels,
        ),
        dim=1,
    )

    global_token_index = (
        prefix_embeddings.shape[1]
        + question_embeddings.shape[1]
    )

    graph_token_index = global_token_index + 1

    reordered_image_start = (
        graph_token_index + 1
    )

    answer_start = (
        reordered_prompt_embeddings.shape[1]
    )

    diagnostics = {
        "original_prompt_length": prompt_length,
        "reordered_prompt_length": (
            reordered_prompt_embeddings.shape[1]
        ),
        "answer_token_count": int(
            answer_attention_mask.sum().item()
        ),
        "training_sequence_length": (
            inputs_embeds.shape[1]
        ),
        "question_token_count": (
            user_turn_end - image_end - 1
        ),
        "global_token_index": (
            global_token_index
        ),
        "graph_token_index": graph_token_index,
        "reordered_image_start": (
            reordered_image_start
        ),
        "answer_start": answer_start,
        "answer_end": (
            answer_start + answer_length
        ),
        "supervised_token_count": int(
            labels.ne(-100).sum().item()
        ),
        "input_order": (
            "question_global_graph_image_answer"
        ),
    }

    if not (
        inputs_embeds.shape[1]
        == attention_mask.shape[1]
        == labels.shape[1]
    ):
        raise RuntimeError(
            "Embedding, mask, and label lengths "
            "do not match."
        )

    return TrainingSequenceBatch(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
        diagnostics=diagnostics,
    )