import types
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from vlm.models.attention_extractor import (
    AttentionExtractionResult,
    SelectiveAttentionCollector,
)
from vlm.models.attention_profile import (
    extract_local_attention_profiles,
)
from vlm.models.attention_reconstructor import (
    ReconstructedAttention,
    reconstruct_local_attention,
)
from vlm.models.global_graph_token_injector import (
    _prepare_visual_input_embeddings,
    chat_with_global_graph_tokens,
)
from vlm.models.global_token_encoder import (
    GlobalTokenEncoder,
    GlobalTokenEncoderOutput,
)
from vlm.models.graph_node_builder import (
    extract_projected_visual_features,
)
from vlm.models.graph_token_encoder import (
    GraphTokenEncoder,
    GraphTokenEncoderOutput,
)
from vlm.models.multimodal_sequence_builder import (
    TrainingSequenceBatch,
    build_training_sequence,
)
from vlm.models.qlora_adapter import (
    disable_language_adapter,
    enable_language_adapter,
    has_qlora_adapter,
)


@dataclass
class FirstPassOutput:
    attention_result: AttentionExtractionResult
    local_attention_profiles: torch.Tensor
    reconstructed_attention: ReconstructedAttention
    projected_visual_features: torch.Tensor
    tile_grid: Tuple[int, int]
    image_size: Tuple[int, int]


@dataclass
class GlobalRelationalGenerationOutput:
    response: str
    first_pass: FirstPassOutput
    global_output: GlobalTokenEncoderOutput
    graph_output: GraphTokenEncoderOutput
    second_pass_diagnostics: Dict



@dataclass
class GlobalRelationalTrainingOutput:
    loss: torch.Tensor
    first_pass: FirstPassOutput
    global_output: GlobalTokenEncoderOutput
    graph_output: GraphTokenEncoderOutput
    sequence_batch: TrainingSequenceBatch
    diagnostics: Dict


def _capture_internvl_prompt_inputs(
    backbone,
    tokenizer,
    pixel_values: torch.Tensor,
    question: str,
):
    """
    Build the exact InternVL chat prompt without
    running language generation.

    InternVL chat() constructs the official prompt
    and calls backbone.generate(). This function
    temporarily replaces generate() with a capture
    function that records input_ids and the mask.
    """

    if not isinstance(question, str):
        raise TypeError(
            "question must be a string."
        )

    if not question.strip():
        raise ValueError(
            "question must not be empty."
        )

    original_generate = backbone.generate
    captured = {}

    def capture_generate(
        self,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        **generate_kwargs,
    ):
        if input_ids is None:
            raise RuntimeError(
                "InternVL chat did not provide "
                "input_ids."
            )

        if attention_mask is None:
            raise RuntimeError(
                "InternVL chat did not provide "
                "attention_mask."
            )

        captured["input_ids"] = (
            input_ids.detach().clone()
        )

        captured["attention_mask"] = (
            attention_mask.detach().clone()
        )

        eos_token_id = generate_kwargs.get(
            "eos_token_id",
            tokenizer.eos_token_id,
        )

        if isinstance(
            eos_token_id,
            (list, tuple),
        ):
            eos_token_id = eos_token_id[0]

        if eos_token_id is None:
            raise RuntimeError(
                "Could not determine the "
                "InternVL turn-end token."
            )

        return torch.full(
            (input_ids.shape[0], 1),
            fill_value=int(eos_token_id),
            dtype=torch.long,
            device=input_ids.device,
        )

    backbone.generate = types.MethodType(
        capture_generate,
        backbone,
    )

    try:
        backbone.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=f"<image>\n{question}",
            generation_config={
                "max_new_tokens": 1,
                "do_sample": False,
                "num_beams": 1,
            },
        )

    finally:
        backbone.generate = original_generate

    if "input_ids" not in captured:
        raise RuntimeError(
            "Failed to capture InternVL "
            "prompt input IDs."
        )

    return (
        captured["input_ids"],
        captured["attention_mask"],
    )

class GlobalRelationalVLM(nn.Module):
    """
    Integrated two-pass Global-Relational VLM.

    Current implementation:
        - First-pass attention extraction
        - Global Token generation
        - Graph Token generation
        - Second-pass response generation

    Training forward:
        - Frozen InternVL backbone
        - Trainable Global and Graph encoders
        - Answer-only teacher-forcing loss
    """

    def __init__(
        self,
        backbone,
        tokenizer,
        layer_indices: Sequence[int] = tuple(
            range(10, 18)
        ),
        tile_size: int = 448,
        global_encoder: Optional[
            GlobalTokenEncoder
        ] = None,
        graph_encoder: Optional[
            GraphTokenEncoder
        ] = None,
    ):
        super().__init__()

        self.backbone = backbone
        self.tokenizer = tokenizer

        self.layer_indices = tuple(
            int(index) for index in layer_indices
        )

        self.tile_size = tile_size

        if global_encoder is None:
            global_encoder = GlobalTokenEncoder(
                map_size=32,
                descriptor_dimension=16,
                feature_dimension=128,
                llm_hidden_dimension=3584,
                top_p=0.05,
                peak_threshold=0.50,
                max_peak_count=16,
                dropout=0.1,
                modulation_scale=0.5,
                initial_token_scale=0.35,
            )

        if graph_encoder is None:
            graph_encoder = GraphTokenEncoder(
                top_k=16,
                token_grid=(16, 16),
                sigma_p=0.25,
                threshold=0.70,
                spatial_weight=0.35,
                visual_weight=0.20,
                attention_weight=0.45,
                node_input_dimension=3587,
                graph_dimension=256,
                gnn_layers=2,
                llm_hidden_dimension=3584,
            )

        self.global_encoder = global_encoder
        self.graph_encoder = graph_encoder

    @property
    def descriptor_statistics_fitted(self):
        return (
            self.global_encoder
            .descriptor_statistics_fitted
        )

    @torch.no_grad()
    def set_descriptor_statistics(
        self,
        mean,
        std,
    ):
        self.global_encoder.set_descriptor_statistics(
            mean=mean,
            std=std,
        )

    def auxiliary_parameter_count(self):
        return sum(
            parameter.numel()
            for module in (
                self.global_encoder,
                self.graph_encoder,
            )
            for parameter in module.parameters()
            if parameter.requires_grad
        )

    def extract_first_pass(
        self,
        pixel_values: torch.Tensor,
        question: str,
        tile_grid: Tuple[int, int],
        image_size: Tuple[int, int],
    ) -> FirstPassOutput:
        tile_grid = tuple(
            int(value) for value in tile_grid
        )

        image_size = tuple(
            int(value) for value in image_size
        )

        local_tile_count = (
            tile_grid[0] * tile_grid[1]
        )

        total_tile_count = int(
            pixel_values.shape[0]
        )

        tokens_per_tile = int(
            self.backbone.num_image_token
        )

        expected_total_tiles = {
            local_tile_count,
            local_tile_count + 1,
        }

        if total_tile_count not in (
            expected_total_tiles
        ):
            raise ValueError(
                "Unexpected total tile count: "
                f"{total_tile_count}. Expected one of "
                f"{sorted(expected_total_tiles)}."
            )

        collector = SelectiveAttentionCollector(
            model=self.backbone,
            tokenizer=self.tokenizer,
            question=question,
            layer_indices=self.layer_indices,
            total_tile_count=total_tile_count,
            local_tile_count=local_tile_count,
            tokens_per_tile=tokens_per_tile,
        )

        backbone_was_training = (
            self.backbone.training
        )

        self.backbone.eval()

        qlora_attached = has_qlora_adapter(
            self.backbone
        )

        if qlora_attached:
            first_pass_adapter_context = (
                disable_language_adapter(
                    self.backbone
                )
            )
        else:
            first_pass_adapter_context = (
                nullcontext()
            )

        try:
            collector.install()

            try:
                with first_pass_adapter_context:
                    with torch.inference_mode():
                        self.backbone.chat(
                            tokenizer=(
                                self.tokenizer
                            ),
                            pixel_values=(
                                pixel_values
                            ),
                            question=(
                                f"<image>\n"
                                f"{question}"
                            ),
                            generation_config={
                                "max_new_tokens": 1,
                                "do_sample": False,
                                "num_beams": 1,
                            },
                        )

            finally:
                collector.remove()

            attention_result = (
                collector.build_result()
            )

            local_attention_profiles = (
                extract_local_attention_profiles(
                    attention_result
                )
            )

            reconstructed_attention = (
                reconstruct_local_attention(
                    local_attention=(
                        attention_result
                        .local_image_l1
                    ),
                    tile_grid=tile_grid,
                    original_size=image_size,
                    tile_size=self.tile_size,
                )
            )

            with torch.inference_mode():
                projected_visual_features = (
                    extract_projected_visual_features(
                        model=self.backbone,
                        pixel_values=pixel_values,
                    )
                )

        finally:
            if backbone_was_training:
                self.backbone.train()
            else:
                self.backbone.eval()

        return FirstPassOutput(
            attention_result=attention_result,
            local_attention_profiles=(
                local_attention_profiles
            ),
            reconstructed_attention=(
                reconstructed_attention
            ),
            projected_visual_features=(
                projected_visual_features
            ),
            tile_grid=tile_grid,
            image_size=image_size,
        )

    def build_auxiliary_tokens(
        self,
        first_pass: FirstPassOutput,
    ):
        global_device = next(
            self.global_encoder.parameters()
        ).device

        attention_map = (
            first_pass
            .reconstructed_attention
            .original_map
            .to(
                device=global_device,
                dtype=torch.float32,
            )
        )

        global_output = self.global_encoder(
            attention_map
        )

        graph_parameter = next(
            self.graph_encoder.parameters()
        )

        graph_device = graph_parameter.device
        graph_dtype = graph_parameter.dtype

        graph_output = self.graph_encoder(
            local_attention=(
                first_pass
                .attention_result
                .local_image_l1
                .to(
                    device=graph_device,
                    dtype=graph_dtype,
                )
            ),
            attention_profiles=(
                first_pass
                .local_attention_profiles
                .to(
                    device=graph_device,
                    dtype=graph_dtype,
                )
            ),
            projected_visual_features=(
                first_pass
                .projected_visual_features
                .to(
                    device=graph_device,
                    dtype=graph_dtype,
                )
            ),
            tile_grid=first_pass.tile_grid,
            image_size=first_pass.image_size,
        )

        return global_output, graph_output

    def freeze_backbone(self):
        """
        Freeze the InternVL base weights.

        When QLoRA is attached, only LoRA
        parameters remain trainable.
        """

        qlora_attached = has_qlora_adapter(
            self.backbone
        )

        adapter_parameters = []

        if qlora_attached:
            adapter_parameters = [
                parameter
                for name, parameter
                in self.backbone
                .language_model
                .named_parameters()
                if "lora_" in name
            ]

            if not adapter_parameters:
                raise RuntimeError(
                    "QLoRA is attached but no "
                    "LoRA parameters were found."
                )

        self.backbone.requires_grad_(False)

        for parameter in adapter_parameters:
            parameter.requires_grad_(True)

        self.backbone.eval()

    def forward_train(
        self,
        pixel_values: torch.Tensor,
        question: str,
        answer: str,
        tile_grid: Tuple[int, int],
        image_size: Tuple[int, int],
        supervise_turn_end: bool = True,
        save_activations_on_cpu: bool = True,
        first_pass: Optional[
            FirstPassOutput
        ] = None,
    ) -> GlobalRelationalTrainingOutput:
        """
        Run one teacher-forcing training forward.

        The frozen backbone supplies First-Pass
        attention and visual features. Only the
        Global and Graph encoders are trainable in
        the Step-23 baseline.
        """

        if not isinstance(answer, str):
            raise TypeError(
                "answer must be a string."
            )

        if not answer.strip():
            raise ValueError(
                "answer must not be empty."
            )

        if not (
            self.global_encoder
            .descriptor_statistics_fitted
        ):
            raise RuntimeError(
                "Final descriptor statistics "
                "must be loaded before training."
            )

        qlora_attached = has_qlora_adapter(
            self.backbone
        )

        trainable_backbone_named_parameters = [
            (name, parameter)
            for name, parameter
            in self.backbone.named_parameters()
            if parameter.requires_grad
        ]

        trainable_backbone_parameters = sum(
            parameter.numel()
            for _, parameter
            in trainable_backbone_named_parameters
        )

        qlora_trainable_parameters = sum(
            parameter.numel()
            for name, parameter
            in trainable_backbone_named_parameters
            if "lora_" in name
        )

        unexpected_trainable_parameters = [
            name
            for name, _
            in trainable_backbone_named_parameters
            if not (
                qlora_attached
                and "lora_" in name
            )
        ]

        if unexpected_trainable_parameters:
            raise RuntimeError(
                "Unexpected trainable backbone "
                "parameters: "
                + ", ".join(
                    unexpected_trainable_parameters[
                        :10
                    ]
                )
            )

        if (
            not qlora_attached
            and trainable_backbone_parameters
            != 0
        ):
            raise RuntimeError(
                "The non-QLoRA backbone must "
                "be fully frozen."
            )

        if (
            qlora_attached
            and qlora_trainable_parameters
            <= 0
        ):
            raise RuntimeError(
                "QLoRA is attached but no "
                "trainable LoRA parameters "
                "were found."
            )

        self.backbone.eval()

        first_pass_was_cached = (
            first_pass is not None
        )

        if first_pass is None:
            first_pass = (
                self.extract_first_pass(
                    pixel_values=pixel_values,
                    question=question,
                    tile_grid=tile_grid,
                    image_size=image_size,
                )
            )
        else:
            expected_tile_grid = tuple(
                int(value)
                for value in tile_grid
            )

            expected_image_size = tuple(
                int(value)
                for value in image_size
            )

            if tuple(
                first_pass.tile_grid
            ) != expected_tile_grid:
                raise ValueError(
                    "Cached First Pass tile grid "
                    "does not match the input."
                )

            if tuple(
                first_pass.image_size
            ) != expected_image_size:
                raise ValueError(
                    "Cached First Pass image size "
                    "does not match the input."
                )

        global_device = next(
            self.global_encoder.parameters()
        ).device

        attention_map = (
            first_pass
            .reconstructed_attention
            .original_map
            .detach()
            .clone()
            .to(
                device=global_device,
                dtype=torch.float32,
            )
        )

        global_output = self.global_encoder(
            attention_map
        )

        graph_parameter = next(
            self.graph_encoder.parameters()
        )

        graph_device = graph_parameter.device
        graph_dtype = graph_parameter.dtype

        local_attention = (
            first_pass
            .attention_result
            .local_image_l1
            .detach()
            .clone()
            .to(
                device=graph_device,
                dtype=graph_dtype,
            )
        )

        attention_profiles = (
            first_pass
            .local_attention_profiles
            .detach()
            .clone()
            .to(
                device=graph_device,
                dtype=graph_dtype,
            )
        )

        graph_visual_features = (
            first_pass
            .projected_visual_features
            .detach()
            .clone()
            .to(
                device=graph_device,
                dtype=graph_dtype,
            )
        )

        graph_output = self.graph_encoder(
            local_attention=local_attention,
            attention_profiles=(
                attention_profiles
            ),
            projected_visual_features=(
                graph_visual_features
            ),
            tile_grid=first_pass.tile_grid,
            image_size=first_pass.image_size,
        )

        (
            prompt_input_ids,
            prompt_attention_mask,
        ) = _capture_internvl_prompt_inputs(
            backbone=self.backbone,
            tokenizer=self.tokenizer,
            pixel_values=pixel_values,
            question=question,
        )

        embedding_layer = (
            self.backbone
            .language_model
            .get_input_embeddings()
        )

        embedding_device = next(
            embedding_layer.parameters()
        ).device

        prompt_input_ids = (
            prompt_input_ids.to(
                device=embedding_device,
                dtype=torch.long,
            )
        )

        prompt_attention_mask = (
            prompt_attention_mask.to(
                device=embedding_device
            )
        )

        prompt_visual_features = (
            first_pass
            .projected_visual_features
            .detach()
            .clone()
        )

        (
            prompt_input_embeddings,
            _,
        ) = _prepare_visual_input_embeddings(
            model=self.backbone,
            input_ids=prompt_input_ids,
            pixel_values=pixel_values,
            visual_features=(
                prompt_visual_features
            ),
        )

        answer_encoding = self.tokenizer(
            answer,
            add_special_tokens=False,
            return_tensors="pt",
        )

        answer_input_ids = (
            answer_encoding["input_ids"]
        )

        if answer_input_ids.numel() == 0:
            raise RuntimeError(
                "The answer produced no tokens."
            )

        turn_end_id = (
            self.tokenizer
            .convert_tokens_to_ids(
                "<|im_end|>"
            )
        )

        if turn_end_id is None:
            raise RuntimeError(
                "Could not find <|im_end|>."
            )

        if supervise_turn_end:
            turn_end_tensor = torch.tensor(
                [[int(turn_end_id)]],
                dtype=torch.long,
            )

            if int(
                answer_input_ids[0, -1].item()
            ) != int(turn_end_id):
                answer_input_ids = torch.cat(
                    (
                        answer_input_ids,
                        turn_end_tensor,
                    ),
                    dim=1,
                )

        answer_input_ids = (
            answer_input_ids.to(
                device=embedding_device,
                dtype=torch.long,
            )
        )

        answer_embeddings = embedding_layer(
            answer_input_ids
        )

        answer_attention_mask = torch.ones(
            answer_input_ids.shape,
            dtype=prompt_attention_mask.dtype,
            device=embedding_device,
        )

        sequence_batch = (
            build_training_sequence(
                tokenizer=self.tokenizer,
                prompt_input_ids=(
                    prompt_input_ids
                ),
                prompt_input_embeddings=(
                    prompt_input_embeddings
                ),
                prompt_attention_mask=(
                    prompt_attention_mask
                ),
                global_token=(
                    global_output.global_token
                ),
                graph_token=(
                    graph_output.graph_token
                ),
                answer_input_ids=(
                    answer_input_ids
                ),
                answer_embeddings=(
                    answer_embeddings
                ),
                answer_attention_mask=(
                    answer_attention_mask
                ),
            )
        )

        adapter_training = bool(
            self.global_encoder.training
            or self.graph_encoder.training
        )

        if qlora_attached:
            enable_language_adapter(
                self.backbone,
                adapter_name=(
                    "global_relational"
                ),
            )

            self.backbone.language_model.train(
                adapter_training
            )
        else:
            self.backbone.language_model.eval()

        if save_activations_on_cpu:
            activation_context = (
                torch.autograd.graph.save_on_cpu(
                    pin_memory=True
                )
            )
        else:
            activation_context = nullcontext()

        with activation_context:
            language_output = (
                self.backbone.language_model(
                    inputs_embeds=(
                        sequence_batch.inputs_embeds
                    ),
                    attention_mask=(
                        sequence_batch.attention_mask
                    ),
                    labels=(
                        sequence_batch.labels
                    ),
                    use_cache=False,
                    return_dict=True,
                )
            )

        if language_output.loss is None:
            raise RuntimeError(
                "Language model did not "
                "return a training loss."
            )

        loss = language_output.loss

        shifted_logits = (
            language_output.logits[
                :,
                :-1,
                :,
            ]
        )

        shifted_labels = (
            sequence_batch.labels[
                :,
                1:,
            ]
        )

        supervised_mask = (
            shifted_labels.ne(-100)
        )

        supervised_token_total = int(
            supervised_mask.sum().item()
        )

        if supervised_token_total == 0:
            raise RuntimeError(
                "No supervised answer tokens "
                "were found after label shifting."
            )

        predicted_ids = (
            shifted_logits
            .detach()
            .argmax(dim=-1)
        )

        supervised_token_correct = int(
            predicted_ids[
                supervised_mask
            ]
            .eq(
                shifted_labels[
                    supervised_mask
                ]
            )
            .sum()
            .item()
        )

        teacher_forced_accuracy = (
            supervised_token_correct
            / supervised_token_total
        )

        diagnostics = dict(
            sequence_batch.diagnostics
        )

        diagnostics.update(
            {
                "loss": float(
                    loss.detach()
                    .float()
                    .item()
                ),
                "logits_shape": tuple(
                    language_output
                    .logits.shape
                ),
                "backbone_trainable_parameters": (
                    trainable_backbone_parameters
                ),
                "qlora_attached": bool(
                    qlora_attached
                ),
                "qlora_trainable_parameters": (
                    qlora_trainable_parameters
                ),
                "unexpected_backbone_trainable": (
                    len(
                        unexpected_trainable_parameters
                    )
                ),
                "adapter_training": bool(
                    adapter_training
                ),
                "global_trainable_parameters": sum(
                    parameter.numel()
                    for parameter
                    in self.global_encoder
                    .parameters()
                    if parameter.requires_grad
                ),
                "graph_trainable_parameters": sum(
                    parameter.numel()
                    for parameter
                    in self.graph_encoder
                    .parameters()
                    if parameter.requires_grad
                ),
                "supervise_turn_end": bool(
                    supervise_turn_end
                ),
                "save_activations_on_cpu": bool(
                    save_activations_on_cpu
                ),
                "first_pass_cached": bool(
                    first_pass_was_cached
                ),
                "teacher_forced_token_correct": (
                    supervised_token_correct
                ),
                "teacher_forced_token_total": (
                    supervised_token_total
                ),
                "teacher_forced_token_accuracy": (
                    teacher_forced_accuracy
                ),
            }
        )

        del language_output

        return GlobalRelationalTrainingOutput(
            loss=loss,
            first_pass=first_pass,
            global_output=global_output,
            graph_output=graph_output,
            sequence_batch=sequence_batch,
            diagnostics=diagnostics,
        )

    @torch.inference_mode()
    def generate(
        self,
        pixel_values: torch.Tensor,
        question: str,
        tile_grid: Tuple[int, int],
        image_size: Tuple[int, int],
        generation_config=None,
        first_pass: Optional[
            FirstPassOutput
        ] = None,
    ) -> GlobalRelationalGenerationOutput:
        self.eval()

        if generation_config is None:
            generation_config = {
                "max_new_tokens": 32,
                "do_sample": False,
                "num_beams": 1,
            }

        first_pass_was_cached = (
            first_pass is not None
        )

        if first_pass is None:
            first_pass = (
                self.extract_first_pass(
                    pixel_values=pixel_values,
                    question=question,
                    tile_grid=tile_grid,
                    image_size=image_size,
                )
            )
        else:
            expected_tile_grid = tuple(
                int(value)
                for value in tile_grid
            )
            expected_image_size = tuple(
                int(value)
                for value in image_size
            )

            if tuple(
                first_pass.tile_grid
            ) != expected_tile_grid:
                raise ValueError(
                    "Cached First Pass tile grid "
                    "does not match the input."
                )

            if tuple(
                first_pass.image_size
            ) != expected_image_size:
                raise ValueError(
                    "Cached First Pass image size "
                    "does not match the input."
                )

        (
            global_output,
            graph_output,
        ) = self.build_auxiliary_tokens(
            first_pass
        )

        qlora_attached = (
            has_qlora_adapter(
                self.backbone
            )
        )

        if qlora_attached:
            enable_language_adapter(
                self.backbone,
                adapter_name=(
                    "global_relational"
                ),
            )

        (
            response,
            second_pass_diagnostics,
        ) = chat_with_global_graph_tokens(
            model=self.backbone,
            tokenizer=self.tokenizer,
            pixel_values=pixel_values,
            question=question,
            global_token=(
                global_output.global_token
            ),
            graph_token=(
                graph_output.graph_token
            ),
            generation_config=generation_config,
            visual_features=(
                first_pass
                .projected_visual_features
            ),
        )

        second_pass_diagnostics = dict(
            second_pass_diagnostics
        )

        second_pass_diagnostics.update(
            {
                "first_pass_cached": bool(
                    first_pass_was_cached
                ),
                "qlora_attached": bool(
                    qlora_attached
                ),
                "adapter_enabled": bool(
                    qlora_attached
                ),
            }
        )

        return GlobalRelationalGenerationOutput(
            response=response,
            first_pass=first_pass,
            global_output=global_output,
            graph_output=graph_output,
            second_pass_diagnostics=(
                second_pass_diagnostics
            ),
        )