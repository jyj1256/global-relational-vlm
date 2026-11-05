from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch


@dataclass
class AttentionExtractionResult:
    layer_indices: List[int]

    # [selected_layers, heads, all_image_tokens]
    per_layer_head: torch.Tensor

    # Image token positions in the complete LLM sequence
    image_positions: List[int]
    local_image_positions: List[int]
    thumbnail_positions: List[int]

    question_start: int
    question_end: int
    last_question_token: int

    # Mean attention before image-only L1 normalization
    mean_all_raw: torch.Tensor
    mean_local_raw: torch.Tensor
    mean_thumbnail_raw: torch.Tensor

    # Image-only distributions
    all_image_l1: torch.Tensor
    local_image_l1: torch.Tensor


def _to_token_id_list(tokenizer, text):
    encoded = tokenizer(
        text,
        add_special_tokens=False,
    ).input_ids

    if (
        isinstance(encoded, list)
        and len(encoded) > 0
        and isinstance(encoded[0], list)
    ):
        encoded = encoded[0]

    return list(encoded)


def _find_all_subsequences(sequence, pattern):
    if not pattern:
        return []

    matches = []
    pattern_length = len(pattern)

    for start in range(
        len(sequence) - pattern_length + 1
    ):
        end = start + pattern_length

        if sequence[start:end] == pattern:
            matches.append((start, end - 1))

    return matches


def _find_question_span(
    tokenizer,
    input_ids,
    question,
):
    variants = [
        question,
        " " + question,
        "\n" + question,
    ]

    candidates = []

    for variant in variants:
        variant_ids = _to_token_id_list(
            tokenizer,
            variant,
        )

        matches = _find_all_subsequences(
            input_ids,
            variant_ids,
        )

        for start, end in matches:
            candidates.append(
                {
                    "start": start,
                    "end": end,
                    "variant": variant,
                }
            )

    if not candidates:
        raise RuntimeError(
            "The user question could not be located "
            "inside the InternVL input sequence."
        )

    return max(
        candidates,
        key=lambda candidate: candidate["end"],
    )


def _l1_normalize(values, epsilon=1e-12):
    denominator = values.sum().clamp_min(epsilon)
    return values / denominator


class SelectiveAttentionCollector:
    """
    Collects only the last-question-token-to-image-token
    attention rows from selected Qwen2 decoder layers.

    InternVL and Transformers source files are not modified.
    """

    def __init__(
        self,
        model,
        tokenizer,
        question: str,
        layer_indices: Sequence[int],
        total_tile_count: int,
        local_tile_count: int,
        tokens_per_tile: int,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.question = question

        self.layer_indices = list(layer_indices)
        self.total_tile_count = total_tile_count
        self.local_tile_count = local_tile_count
        self.tokens_per_tile = tokens_per_tile

        self.image_context_token_id = (
            tokenizer.convert_tokens_to_ids(
                "<IMG_CONTEXT>"
            )
        )

        self.input_ids = None
        self.image_positions = None
        self.local_image_positions = None
        self.thumbnail_positions = None

        self.question_start = None
        self.question_end = None
        self.last_question_token = None

        self.captured: Dict[int, torch.Tensor] = {}
        self.hook_handles = []
        self.original_generate = None
        self.installed = False

    def _prepare_sequence_information(
        self,
        input_ids_tensor,
    ):
        if input_ids_tensor.ndim != 2:
            raise ValueError(
                "Expected input_ids with shape "
                "[batch, sequence]."
            )

        if input_ids_tensor.shape[0] != 1:
            raise ValueError(
                "Attention inspection currently supports "
                "batch size 1 only."
            )

        input_ids = (
            input_ids_tensor[0]
            .detach()
            .cpu()
            .tolist()
        )

        image_positions = [
            index
            for index, token_id in enumerate(input_ids)
            if token_id == self.image_context_token_id
        ]

        expected_total = (
            self.total_tile_count
            * self.tokens_per_tile
        )

        if len(image_positions) != expected_total:
            raise RuntimeError(
                "Image token count mismatch: "
                f"expected {expected_total}, "
                f"found {len(image_positions)}."
            )

        expected_local = (
            self.local_tile_count
            * self.tokens_per_tile
        )

        question_span = _find_question_span(
            tokenizer=self.tokenizer,
            input_ids=input_ids,
            question=self.question,
        )

        self.input_ids = input_ids
        self.image_positions = image_positions

        self.local_image_positions = (
            image_positions[:expected_local]
        )
        self.thumbnail_positions = (
            image_positions[expected_local:]
        )

        self.question_start = question_span["start"]
        self.question_end = question_span["end"]
        self.last_question_token = question_span["end"]

    def _wrapped_generate(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids")

        if input_ids is None:
            raise RuntimeError(
                "InternVL generate() did not receive input_ids."
            )

        self._prepare_sequence_information(input_ids)

        pixel_values = kwargs.get(
            "pixel_values"
        )

        visual_features = kwargs.get(
            "visual_features"
        )

        if pixel_values is not None:
            if visual_features is None:
                with torch.inference_mode():
                    visual_features = (
                        self.model
                        .extract_feature(
                            pixel_values
                        )
                    )

            embedding_layer = (
                self.model
                .language_model
                .get_input_embeddings()
            )

            embedding_weight = (
                embedding_layer.weight
            )

            visual_features = (
                visual_features
                .detach()
                .to(
                    device=(
                        embedding_weight.device
                    ),
                    dtype=(
                        embedding_weight.dtype
                    ),
                )
            )

            kwargs["visual_features"] = (
                visual_features
            )

        return self.original_generate(
            *args,
            **kwargs,
        )

    def _make_attention_hook(self, layer_index):
        def hook(_module, _inputs, output):
            # Generation invokes attention again for generated
            # tokens. Keep only the first full prompt pass.
            if layer_index in self.captured:
                return

            if self.last_question_token is None:
                return

            if not isinstance(output, (tuple, list)):
                return

            if len(output) < 2:
                return

            attention_weights = output[1]

            if attention_weights is None:
                return

            if attention_weights.ndim != 4:
                raise RuntimeError(
                    "Expected attention shape "
                    "[batch, heads, query, key], "
                    f"received {tuple(attention_weights.shape)}."
                )

            query_length = attention_weights.shape[-2]
            key_length = attention_weights.shape[-1]

            # Subsequent generation calls usually have a query
            # length of one and must not be collected.
            if self.last_question_token >= query_length:
                return

            if max(self.image_positions) >= key_length:
                return

            key_indices = torch.tensor(
                self.image_positions,
                dtype=torch.long,
                device=attention_weights.device,
            )

            selected_row = attention_weights[
                0,
                :,
                self.last_question_token,
                :,
            ]

            selected_image_attention = (
                selected_row.index_select(
                    dim=-1,
                    index=key_indices,
                )
            )

            self.captured[layer_index] = (
                selected_image_attention
                .detach()
                .to(
                    device="cpu",
                    dtype=torch.float32,
                )
                .contiguous()
            )

        return hook

    def install(self):
        if self.installed:
            raise RuntimeError(
                "Attention collector is already installed."
            )

        language_model = (
            self.model.language_model
        )

        get_base_model = getattr(
            language_model,
            "get_base_model",
            None,
        )

        if callable(get_base_model):
            language_model = (
                get_base_model()
            )

        decoder_model = getattr(
            language_model,
            "model",
            None,
        )

        language_layers = getattr(
            decoder_model,
            "layers",
            None,
        )

        if language_layers is None:
            language_layers = getattr(
                language_model,
                "layers",
                None,
            )

        if language_layers is None:
            raise RuntimeError(
                "Could not locate the language "
                "model decoder layers."
            )

        layer_count = len(language_layers)

        for layer_index in self.layer_indices:
            if not 0 <= layer_index < layer_count:
                raise ValueError(
                    f"Invalid layer index {layer_index}; "
                    f"model has {layer_count} layers."
                )

            attention_module = (
                language_layers[layer_index].self_attn
            )

            handle = (
                attention_module.register_forward_hook(
                    self._make_attention_hook(
                        layer_index
                    )
                )
            )

            self.hook_handles.append(handle)

        self.original_generate = self.model.generate
        self.model.generate = self._wrapped_generate
        self.installed = True

    def remove(self):
        for handle in self.hook_handles:
            handle.remove()

        self.hook_handles.clear()

        if self.original_generate is not None:
            self.model.generate = (
                self.original_generate
            )

        self.installed = False

    def build_result(self):
        missing_layers = [
            layer_index
            for layer_index in self.layer_indices
            if layer_index not in self.captured
        ]

        if missing_layers:
            raise RuntimeError(
                "Attention was not captured from layers: "
                f"{missing_layers}"
            )

        per_layer_head = torch.stack(
            [
                self.captured[layer_index]
                for layer_index in self.layer_indices
            ],
            dim=0,
        )

        local_token_count = len(
            self.local_image_positions
        )

        local_per_layer_head = (
            per_layer_head[
                ...,
                :local_token_count,
            ]
        )

        thumbnail_per_layer_head = (
            per_layer_head[
                ...,
                local_token_count:,
            ]
        )

        mean_all_raw = per_layer_head.mean(
            dim=(0, 1)
        )

        mean_local_raw = (
            local_per_layer_head.mean(
                dim=(0, 1)
            )
        )

        if thumbnail_per_layer_head.shape[-1] > 0:
            mean_thumbnail_raw = (
                thumbnail_per_layer_head.mean(
                    dim=(0, 1)
                )
            )
        else:
            mean_thumbnail_raw = torch.empty(
                0,
                dtype=torch.float32,
            )

        all_image_l1 = _l1_normalize(
            mean_all_raw
        )

        local_image_l1 = _l1_normalize(
            mean_local_raw
        )

        return AttentionExtractionResult(
            layer_indices=self.layer_indices,
            per_layer_head=per_layer_head,
            image_positions=self.image_positions,
            local_image_positions=(
                self.local_image_positions
            ),
            thumbnail_positions=(
                self.thumbnail_positions
            ),
            question_start=self.question_start,
            question_end=self.question_end,
            last_question_token=(
                self.last_question_token
            ),
            mean_all_raw=mean_all_raw,
            mean_local_raw=mean_local_raw,
            mean_thumbnail_raw=(
                mean_thumbnail_raw
            ),
            all_image_l1=all_image_l1,
            local_image_l1=local_image_l1,
        )