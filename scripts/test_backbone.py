import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run single-image inference with InternVL3-8B."
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to a local image.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="What animal is shown in this image?",
        help="Question to ask about the image.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="OpenGVLab/InternVL3-8B",
        help="Hugging Face model ID or local model path.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/home/dm-magneto/dm_yunjin/pretrained/huggingface",
        help="Hugging Face model cache directory.",
    )
    parser.add_argument(
        "--offload-dir",
        type=str,
        default="/home/dm-magneto/dm_yunjin/vlm/outputs/offload",
        help="Directory used for model offloading.",
    )
    parser.add_argument(
        "--gpu-memory",
        type=str,
        default="10GiB",
        help="Maximum GPU memory used by device_map.",
    )
    parser.add_argument(
        "--cpu-memory",
        type=str,
        default="48GiB",
        help="Maximum CPU memory used by device_map.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=448,
        help="InternVL input tile size.",
    )
    parser.add_argument(
        "--max-num",
        type=int,
        default=6,
        help="Maximum number of local image tiles.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of generated answer tokens.",
    )

    return parser.parse_args()


def build_transform(image_size):
    return transforms.Compose(
        [
            transforms.Lambda(
                lambda image: image.convert("RGB")
                if image.mode != "RGB"
                else image
            ),
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def choose_grid(width, height, min_num, max_num, image_size):
    aspect_ratio = width / height

    candidate_grids = set()

    for blocks in range(min_num, max_num + 1):
        for columns in range(1, blocks + 1):
            for rows in range(1, blocks + 1):
                if min_num <= columns * rows <= max_num:
                    candidate_grids.add((columns, rows))

    candidate_grids = sorted(
        candidate_grids,
        key=lambda grid: grid[0] * grid[1],
    )

    best_grid = (1, 1)
    best_difference = float("inf")
    image_area = width * height

    for columns, rows in candidate_grids:
        grid_ratio = columns / rows
        difference = abs(aspect_ratio - grid_ratio)

        if difference < best_difference:
            best_difference = difference
            best_grid = (columns, rows)

        elif difference == best_difference:
            target_area = (
                image_size
                * image_size
                * columns
                * rows
            )

            if image_area > 0.5 * target_area:
                best_grid = (columns, rows)

    return best_grid


def dynamic_preprocess(
    image,
    min_num=1,
    max_num=6,
    image_size=448,
    use_thumbnail=True,
):
    original_width, original_height = image.size

    columns, rows = choose_grid(
        width=original_width,
        height=original_height,
        min_num=min_num,
        max_num=max_num,
        image_size=image_size,
    )

    target_width = image_size * columns
    target_height = image_size * rows

    resized_image = image.resize(
        (target_width, target_height),
        resample=Image.Resampling.BICUBIC,
    )

    tiles = []

    for row in range(rows):
        for column in range(columns):
            left = column * image_size
            upper = row * image_size
            right = left + image_size
            lower = upper + image_size

            tile = resized_image.crop(
                (left, upper, right, lower)
            )
            tiles.append(tile)

    if use_thumbnail and len(tiles) > 1:
        thumbnail = image.resize(
            (image_size, image_size),
            resample=Image.Resampling.BICUBIC,
        )
        tiles.append(thumbnail)

    return tiles, (columns, rows)


def load_image(image_path, image_size, max_num):
    image = Image.open(image_path).convert("RGB")

    tiles, tile_grid = dynamic_preprocess(
        image=image,
        min_num=1,
        max_num=max_num,
        image_size=image_size,
        use_thumbnail=True,
    )

    transform = build_transform(image_size)

    pixel_values = torch.stack(
        [transform(tile) for tile in tiles]
    )

    return image, pixel_values, tile_grid


def get_vision_device(model):
    if hasattr(model, "vision_model"):
        return next(model.vision_model.parameters()).device

    if hasattr(model, "device"):
        return model.device

    return next(model.parameters()).device


def print_device_map(model):
    device_map = getattr(model, "hf_device_map", None)

    if device_map is None:
        print("Model device map: not available")
        return

    print("\nModel device map:")

    for module_name, device in device_map.items():
        print(f"  {module_name or '<root>'}: {device}")


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

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU was not detected by PyTorch."
        )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir=str(cache_dir),
        use_fast=False,
    )

    print("Loading InternVL3-8B...")
    print("The first run may download about 16 GB of model files.")

    max_memory = {
        0: args.gpu_memory,
        "cpu": args.cpu_memory,
    }

    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir=str(cache_dir),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory=max_memory,
        offload_folder=str(offload_dir),
        offload_state_dict=True,
        use_flash_attn=False,
        attn_implementation="eager",
    ).eval()

    print_device_map(model)

    original_image, pixel_values, tile_grid = load_image(
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
    has_thumbnail = pixel_values.shape[0] > local_tile_count

    print("\nInput information:")
    print(f"  Image path: {image_path}")
    print(f"  Original size: {original_image.size}")
    print(f"  Local tile grid: {tile_grid}")
    print(f"  Local tiles: {local_tile_count}")
    print(f"  Global thumbnail: {has_thumbnail}")
    print(f"  Pixel tensor shape: {tuple(pixel_values.shape)}")
    print(f"  Vision device: {vision_device}")
    print(f"  Question: {args.question}")

    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
    }

    question = f"<image>\n{args.question}"

    print("\nGenerating response...")

    with torch.inference_mode():
        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=question,
            generation_config=generation_config,
        )

    print("\nInternVL3 response:")
    print(response)

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)

        print("\nCUDA memory after inference:")
        print(f"  Allocated: {allocated:.2f} GiB")
        print(f"  Reserved: {reserved:.2f} GiB")


if __name__ == "__main__":
    main()