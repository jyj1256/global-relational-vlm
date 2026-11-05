import argparse
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from test_backbone import (
    get_vision_device,
    load_image,
    print_device_map,
)


class CaptureTokenizer:
    """
    Proxy around the original tokenizer.

    It records the exact prompt and input_ids created inside
    InternVL's chat() method without modifying InternVL source code.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, *args, **kwargs):
        output = self.tokenizer(*args, **kwargs)

        if len(args) > 0:
            text_input = args[0]
        else:
            text_input = kwargs.get("text", None)

        input_ids = output.get("input_ids", None)

        if input_ids is not None:
            if isinstance(input_ids, torch.Tensor):
                saved_input_ids = input_ids.detach().cpu().clone()
            else:
                saved_input_ids = input_ids

            self.calls.append(
                {
                    "text": text_input,
                    "input_ids": saved_input_ids,
                    "kwargs": dict(kwargs),
                }
            )

        return output


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect InternVL3 input token locations."
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--question",
        type=str,
        default="What animal is shown on the candy?",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="OpenGVLab/InternVL3-8B",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/home/dm-magneto/dm_yunjin/pretrained/huggingface",
    )
    parser.add_argument(
        "--offload-dir",
        type=str,
        default="/home/dm-magneto/dm_yunjin/vlm/outputs/offload",
    )
    parser.add_argument(
        "--gpu-memory",
        type=str,
        default="10GiB",
    )
    parser.add_argument(
        "--cpu-memory",
        type=str,
        default="48GiB",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=448,
    )
    parser.add_argument(
        "--max-num",
        type=int,
        default=4,
    )

    return parser.parse_args()


def find_all_subsequences(sequence, pattern):
    if len(pattern) == 0:
        return []

    matches = []
    pattern_length = len(pattern)

    for start in range(
        0,
        len(sequence) - pattern_length + 1,
    ):
        end = start + pattern_length

        if sequence[start:end] == pattern:
            matches.append((start, end - 1))

    return matches


def find_consecutive_ranges(positions):
    if not positions:
        return []

    ranges = []
    start = positions[0]
    previous = positions[0]

    for position in positions[1:]:
        if position == previous + 1:
            previous = position
            continue

        ranges.append((start, previous))
        start = position
        previous = position

    ranges.append((start, previous))

    return ranges


def select_prompt_call(calls, question):
    candidates = []

    for call in calls:
        text_input = call["text"]

        if isinstance(text_input, str):
            texts = [text_input]
        elif isinstance(text_input, (list, tuple)):
            texts = [
                text
                for text in text_input
                if isinstance(text, str)
            ]
        else:
            texts = []

        if any(question in text for text in texts):
            candidates.append(call)

    if not candidates:
        raise RuntimeError(
            "Could not find the tokenizer call containing "
            "the user question."
        )

    return candidates[-1]


def extract_single_input_ids(captured_input_ids):
    if isinstance(captured_input_ids, torch.Tensor):
        if captured_input_ids.ndim == 2:
            if captured_input_ids.shape[0] != 1:
                raise ValueError(
                    "Expected batch size 1 during token inspection."
                )

            return captured_input_ids[0].tolist()

        if captured_input_ids.ndim == 1:
            return captured_input_ids.tolist()

    if (
        isinstance(captured_input_ids, list)
        and len(captured_input_ids) > 0
        and isinstance(captured_input_ids[0], list)
    ):
        if len(captured_input_ids) != 1:
            raise ValueError(
                "Expected batch size 1 during token inspection."
            )

        return captured_input_ids[0]

    return list(captured_input_ids)


def find_question_span(tokenizer, input_ids, question):
    text_variants = [
        question,
        " " + question,
        "\n" + question,
    ]

    candidates = []

    for variant in text_variants:
        variant_ids = tokenizer(
            variant,
            add_special_tokens=False,
        ).input_ids

        matches = find_all_subsequences(
            sequence=input_ids,
            pattern=variant_ids,
        )

        for start, end in matches:
            candidates.append(
                {
                    "start": start,
                    "end": end,
                    "variant": variant,
                    "ids": variant_ids,
                }
            )

    if not candidates:
        return None

    # The user question normally appears at the last matching
    # location before the assistant-generation section.
    return max(
        candidates,
        key=lambda candidate: candidate["end"],
    )


def print_token_window(
    tokenizer,
    input_ids,
    center,
    radius=5,
    image_context_id=None,
    question_span=None,
):
    start = max(0, center - radius)
    end = min(len(input_ids), center + radius + 1)

    print(f"\nToken window around index {center}:")

    for index in range(start, end):
        token_id = input_ids[index]
        token_text = tokenizer.convert_ids_to_tokens(token_id)
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
        )

        labels = []

        if token_id == image_context_id:
            labels.append("IMAGE_CONTEXT")

        if question_span is not None:
            if question_span["start"] <= index <= question_span["end"]:
                labels.append("QUESTION")

            if index == question_span["end"]:
                labels.append("LAST_QUESTION_TOKEN")

        label_text = ", ".join(labels)

        print(
            f"  [{index:5d}] "
            f"id={token_id:7d} "
            f"token={token_text!r} "
            f"decoded={decoded!r} "
            f"{label_text}"
        )


def print_llm_configuration(model):
    llm_config = getattr(
        model.config,
        "llm_config",
        None,
    )

    if llm_config is None:
        llm_config = getattr(
            model.language_model,
            "config",
            None,
        )

    print("\nLLM configuration:")

    if llm_config is None:
        print("  LLM configuration was not found.")
        return

    print(
        "  Hidden size:",
        getattr(llm_config, "hidden_size", "unknown"),
    )
    print(
        "  Hidden layers:",
        getattr(
            llm_config,
            "num_hidden_layers",
            "unknown",
        ),
    )
    print(
        "  Attention heads:",
        getattr(
            llm_config,
            "num_attention_heads",
            "unknown",
        ),
    )
    print(
        "  Key-value heads:",
        getattr(
            llm_config,
            "num_key_value_heads",
            "unknown",
        ),
    )


def main():
    args = parse_args()

    image_path = Path(args.image)
    cache_dir = Path(args.cache_dir)
    offload_dir = Path(args.offload_dir)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    offload_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...")

    base_tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir=str(cache_dir),
        use_fast=False,
    )

    capture_tokenizer = CaptureTokenizer(base_tokenizer)

    print("Loading InternVL3-8B...")

    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir=str(cache_dir),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory={
            0: args.gpu_memory,
            "cpu": args.cpu_memory,
        },
        offload_folder=str(offload_dir),
        offload_state_dict=True,
        use_flash_attn=False,
        attn_implementation="eager",
    ).eval()

    print_device_map(model)
    print_llm_configuration(model)

    _, pixel_values, tile_grid = load_image(
        image_path=image_path,
        image_size=args.image_size,
        max_num=args.max_num,
    )

    vision_device = get_vision_device(model)

    pixel_values = pixel_values.to(
        device=vision_device,
        dtype=torch.bfloat16,
    )

    local_tile_count = tile_grid[0] * tile_grid[1]
    total_tile_count = pixel_values.shape[0]
    has_thumbnail = total_tile_count > local_tile_count

    question_with_image = (
        f"<image>\n{args.question}"
    )

    generation_config = {
        "max_new_tokens": 8,
        "do_sample": False,
        "num_beams": 1,
    }

    print("\nRunning chat() to capture its exact input tokens...")

    with torch.inference_mode():
        response = model.chat(
            tokenizer=capture_tokenizer,
            pixel_values=pixel_values,
            question=question_with_image,
            generation_config=generation_config,
        )

    captured_call = select_prompt_call(
        calls=capture_tokenizer.calls,
        question=args.question,
    )

    input_ids = extract_single_input_ids(
        captured_call["input_ids"]
    )

    image_context_token = "<IMG_CONTEXT>"
    image_start_token = "<img>"
    image_end_token = "</img>"

    image_context_id = base_tokenizer.convert_tokens_to_ids(
        image_context_token
    )
    image_start_id = base_tokenizer.convert_tokens_to_ids(
        image_start_token
    )
    image_end_id = base_tokenizer.convert_tokens_to_ids(
        image_end_token
    )

    image_positions = [
        index
        for index, token_id in enumerate(input_ids)
        if token_id == image_context_id
    ]

    image_ranges = find_consecutive_ranges(
        image_positions
    )

    question_span = find_question_span(
        tokenizer=base_tokenizer,
        input_ids=input_ids,
        question=args.question,
    )

    tokens_per_tile = getattr(
        model,
        "num_image_token",
        None,
    )

    if tokens_per_tile is None and total_tile_count > 0:
        if len(image_positions) % total_tile_count == 0:
            tokens_per_tile = (
                len(image_positions) // total_tile_count
            )

    expected_image_tokens = None

    if tokens_per_tile is not None:
        expected_image_tokens = (
            tokens_per_tile * total_tile_count
        )

    print("\nSpecial image tokens:")
    print(
        f"  {image_start_token}: {image_start_id}"
    )
    print(
        f"  {image_context_token}: {image_context_id}"
    )
    print(
        f"  {image_end_token}: {image_end_id}"
    )

    print("\nInput summary:")
    print(f"  Total sequence length: {len(input_ids)}")
    print(f"  Local tile grid: {tile_grid}")
    print(f"  Local tile count: {local_tile_count}")
    print(f"  Global thumbnail: {has_thumbnail}")
    print(f"  Total image inputs: {total_tile_count}")
    print(f"  Image context tokens: {len(image_positions)}")
    print(f"  Tokens per tile: {tokens_per_tile}")
    print(
        f"  Expected image tokens: "
        f"{expected_image_tokens}"
    )
    print(f"  Image token ranges: {image_ranges}")

    if expected_image_tokens is not None:
        if len(image_positions) == expected_image_tokens:
            print("  Image token count check: PASS")
        else:
            print("  Image token count check: FAIL")

    if image_positions:
        print(
            f"  Image token start: {image_positions[0]}"
        )
        print(
            f"  Image token end: {image_positions[-1]}"
        )

    if question_span is None:
        print("\nQuestion token span was not found.")
    else:
        question_ids = input_ids[
            question_span["start"]:
            question_span["end"] + 1
        ]

        decoded_question = base_tokenizer.decode(
            question_ids,
            skip_special_tokens=False,
        )

        print("\nQuestion token information:")
        print(
            f"  Question start: "
            f"{question_span['start']}"
        )
        print(
            f"  Question end: "
            f"{question_span['end']}"
        )
        print(
            f"  Last question token index: "
            f"{question_span['end']}"
        )
        print(
            f"  Last question token ID: "
            f"{input_ids[question_span['end']]}"
        )
        print(
            f"  Decoded question span: "
            f"{decoded_question!r}"
        )

        print_token_window(
            tokenizer=base_tokenizer,
            input_ids=input_ids,
            center=question_span["end"],
            radius=6,
            image_context_id=image_context_id,
            question_span=question_span,
        )

    if image_positions:
        print_token_window(
            tokenizer=base_tokenizer,
            input_ids=input_ids,
            center=image_positions[0],
            radius=3,
            image_context_id=image_context_id,
            question_span=question_span,
        )

        print_token_window(
            tokenizer=base_tokenizer,
            input_ids=input_ids,
            center=image_positions[-1],
            radius=3,
            image_context_id=image_context_id,
            question_span=question_span,
        )

    if (
        tokens_per_tile is not None
        and len(image_positions) == expected_image_tokens
    ):
        print("\nTile-to-token mapping:")

        for tile_index in range(total_tile_count):
            start_offset = tile_index * tokens_per_tile
            end_offset = start_offset + tokens_per_tile - 1

            token_start = image_positions[start_offset]
            token_end = image_positions[end_offset]

            if (
                has_thumbnail
                and tile_index == total_tile_count - 1
            ):
                tile_type = "global thumbnail"
            else:
                row = tile_index // tile_grid[0]
                column = tile_index % tile_grid[0]
                tile_type = (
                    f"local tile row={row}, column={column}"
                )

            print(
                f"  Tile {tile_index:2d}: "
                f"tokens [{token_start}, {token_end}] "
                f"({tile_type})"
            )

    print("\nInternVL3 response:")
    print(response)

    print("\nCUDA memory:")
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)

    print(f"  Allocated: {allocated:.2f} GiB")
    print(f"  Reserved: {reserved:.2f} GiB")


if __name__ == "__main__":
    main()