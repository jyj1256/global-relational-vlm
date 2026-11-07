import math
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from vlm.models.global_relational_vlm import (
    GlobalRelationalVLM,
)
from vlm.models.object_graph_v1.encoder import (
    ObjectGraphTokenEncoderV1Output,
)
from vlm.models.qlora_adapter import (
    attach_qlora_adapter,
    get_trainable_adapter_parameters,
)
from vlm.scripts.test_backbone import (
    get_vision_device,
    load_image,
)
from vlm.utils.descriptor_statistics import (
    load_descriptor_statistics,
)
from vlm.utils.object_detection_cache import (
    load_object_detection_cache,
)


MODEL_NAME = "OpenGVLab/InternVL3-8B"

CACHE_DIR = (
    PROJECT_ROOT
    / "pretrained/huggingface"
)

OFFLOAD_DIR = (
    PROJECT_ROOT
    / "pretrained/offload"
)

IMAGE_PATH = (
    PROJECT_ROOT
    / "vlm/outputs/paper_figures/"
    "baseball_2337002/input.jpg"
)

DETECTION_PATH = (
    PROJECT_ROOT
    / "vlm/outputs/object_detections/"
    "yolo11s_coco/entries/2337002.json"
)

STATISTICS_PATH = (
    PROJECT_ROOT
    / "vlm/configs/"
    "gqa_descriptor_statistics_"
    "nf4_qlora_5000_seed2026.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "vlm/outputs/checkpoints/"
    "object_graph_v1_training_smoke.pt"
)

QUESTION = (
    "What is the man to the left "
    "of the people holding?"
)

ANSWER = "bat"


def gradient_norm(parameters):
    squared_sum = 0.0

    for parameter in parameters:
        if parameter.grad is None:
            continue

        gradient = (
            parameter.grad
            .detach()
            .float()
        )

        squared_sum += float(
            gradient
            .square()
            .sum()
            .item()
        )

    return math.sqrt(squared_sum)


def clone_parameters(parameters):
    return [
        parameter
        .detach()
        .float()
        .cpu()
        .clone()
        for parameter in parameters
    ]


def parameter_delta_norm(
    parameters,
    original_parameters,
):
    squared_sum = 0.0

    for parameter, original in zip(
        parameters,
        original_parameters,
    ):
        difference = (
            parameter
            .detach()
            .float()
            .cpu()
            - original
        )

        squared_sum += float(
            difference
            .square()
            .sum()
            .item()
        )

    return math.sqrt(squared_sum)


def main():
    for required_path in (
        IMAGE_PATH,
        DETECTION_PATH,
        STATISTICS_PATH,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(
                required_path
            )

    torch.manual_seed(2026)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    detection_result = (
        load_object_detection_cache(
            input_path=DETECTION_PATH,
            expected_image_id="2337002",
        )
    )

    detections = detection_result.detections

    print("Object Graph v1 training sample:")
    print("  Image:", IMAGE_PATH)
    print("  Question:", QUESTION)
    print("  Answer:", ANSWER)
    print("  Detections:", len(detections))
    print(
        "  Detection classes:",
        dict(
            Counter(
                detection.class_name
                for detection in detections
            )
        ),
    )

    (
        original_image,
        pixel_values,
        tile_grid,
    ) = load_image(
        image_path=IMAGE_PATH,
        image_size=448,
        max_num=1,
    )

    if tuple(
        detection_result.image_size
    ) != tuple(
        original_image.size
    ):
        raise RuntimeError(
            "Detection image size does not "
            "match the input image."
        )

    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=(
                torch.bfloat16
            ),
            llm_int8_skip_modules=[
                "vision_model",
                "mlp1",
            ],
        )
    )

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        cache_dir=str(CACHE_DIR),
        use_fast=False,
        local_files_only=True,
    )

    print("Loading NF4 InternVL3-8B...")

    backbone = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        cache_dir=str(CACHE_DIR),
        torch_dtype=torch.bfloat16,
        quantization_config=(
            quantization_config
        ),
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory={
            0: "10GiB",
            "cpu": "48GiB",
        },
        offload_folder=str(OFFLOAD_DIR),
        use_flash_attn=False,
        attn_implementation="eager",
        local_files_only=True,
    ).eval()

    print("Attaching QLoRA Adapter...")

    attach_qlora_adapter(
        backbone=backbone,
        rank=8,
        alpha=16,
        dropout=0.05,
        target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ),
        adapter_name="global_relational",
    )

    integrated_model = GlobalRelationalVLM(
        backbone=backbone,
        tokenizer=tokenizer,
        layer_indices=tuple(
            range(10, 18)
        ),
        tile_size=448,
        graph_version="v1",
    )

    vision_device = get_vision_device(
        backbone
    )

    integrated_model.global_encoder.to(
        device=vision_device,
        dtype=torch.float32,
    )

    integrated_model.graph_encoder.to(
        device=vision_device,
        dtype=torch.float32,
    )

    statistics = load_descriptor_statistics(
        STATISTICS_PATH
    )

    if (
        statistics["metadata"].get(
            "first_pass_backend"
        )
        != "nf4-qlora"
    ):
        raise RuntimeError(
            "Expected NF4 QLoRA descriptor "
            "statistics."
        )

    integrated_model.set_descriptor_statistics(
        mean=statistics["mean"],
        std=statistics["std"],
    )

    integrated_model.freeze_backbone()

    integrated_model.global_encoder.train()
    integrated_model.graph_encoder.train()

    pixel_values = pixel_values.to(
        device=vision_device,
        dtype=torch.bfloat16,
    )

    adapter_parameters = list(
        get_trainable_adapter_parameters(
            backbone
        )
    )

    global_parameters = [
        parameter
        for parameter
        in integrated_model
        .global_encoder
        .parameters()
        if parameter.requires_grad
    ]

    graph_parameters = [
        parameter
        for parameter
        in integrated_model
        .graph_encoder
        .parameters()
        if parameter.requires_grad
    ]

    trainable_parameters = (
        global_parameters
        + graph_parameters
        + adapter_parameters
    )

    total_trainable_count = sum(
        parameter.numel()
        for parameter
        in trainable_parameters
    )

    print("\nTrainable parameters:")
    print(
        "  Global:",
        f"{sum(p.numel() for p in global_parameters):,}",
    )
    print(
        "  Object Graph v1:",
        f"{sum(p.numel() for p in graph_parameters):,}",
    )
    print(
        "  QLoRA:",
        f"{sum(p.numel() for p in adapter_parameters):,}",
    )
    print(
        "  Total:",
        f"{total_trainable_count:,}",
    )

    print(
        "\nComputing Adapter-OFF First Pass..."
    )

    first_pass = (
        integrated_model.extract_first_pass(
            pixel_values=pixel_values,
            question=QUESTION,
            tile_grid=tile_grid,
            image_size=original_image.size,
        )
    )

    extraction_calls = {
        "count": 0,
    }

    def forbidden_first_pass(*args, **kwargs):
        extraction_calls["count"] += 1

        raise RuntimeError(
            "First Pass was recomputed despite "
            "the supplied FirstPassOutput."
        )

    integrated_model.extract_first_pass = (
        forbidden_first_pass
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": global_parameters,
                "lr": 1e-4,
            },
            {
                "params": graph_parameters,
                "lr": 1e-4,
            },
            {
                "params": adapter_parameters,
                "lr": 5e-5,
            },
        ],
        weight_decay=0.0,
    )

    global_before = clone_parameters(
        global_parameters
    )
    graph_before = clone_parameters(
        graph_parameters
    )
    adapter_before = clone_parameters(
        adapter_parameters
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    print(
        "Running Object Graph v1 "
        "training forward..."
    )

    training_output = (
        integrated_model.forward_train(
            pixel_values=pixel_values,
            question=QUESTION,
            answer=ANSWER,
            tile_grid=tile_grid,
            image_size=original_image.size,
            supervise_turn_end=True,
            save_activations_on_cpu=True,
            first_pass=first_pass,
            detections=detections,
        )
    )

    if not isinstance(
        training_output.graph_output,
        ObjectGraphTokenEncoderV1Output,
    ):
        raise TypeError(
            "Expected Object Graph v1 output."
        )

    print("Running backward...")

    training_output.loss.backward()

    global_gradient = gradient_norm(
        global_parameters
    )
    graph_gradient = gradient_norm(
        graph_parameters
    )
    adapter_gradient = gradient_norm(
        adapter_parameters
    )

    clipped_gradient = float(
        torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=1.0,
        ).item()
    )

    adapter_parameter_ids = {
        id(parameter)
        for parameter in adapter_parameters
    }

    base_llm_has_gradient = any(
        parameter.grad is not None
        for parameter
        in backbone.language_model.parameters()
        if id(parameter)
        not in adapter_parameter_ids
    )

    vision_or_mlp1_has_gradient = any(
        parameter.grad is not None
        for module in (
            backbone.vision_model,
            backbone.mlp1,
        )
        for parameter in module.parameters()
    )

    optimizer.step()

    global_delta = parameter_delta_norm(
        global_parameters,
        global_before,
    )
    graph_delta = parameter_delta_norm(
        graph_parameters,
        graph_before,
    )
    adapter_delta = parameter_delta_norm(
        adapter_parameters,
        adapter_before,
    )

    graph_output = (
        training_output.graph_output
    )

    nodes = graph_output.hybrid_output.nodes
    edges = graph_output.graph_output.edges

    source_counts = Counter(
        node.source
        for node in nodes
    )

    class_counts = Counter(
        node.class_name
        for node in nodes
    )

    correct_bat_nodes = [
        node
        for node in nodes
        if (
            node.class_name
            == "baseball bat"
            and node.detection_id == 4
        )
    ]

    correct_bat_fused = bool(
        len(correct_bat_nodes) == 1
        and correct_bat_nodes[0].source
        == "fused"
    )

    if correct_bat_nodes:
        correct_bat_degree = int(
            graph_output
            .graph_output
            .degree[
                correct_bat_nodes[0]
                .node_id
            ]
            .item()
        )
    else:
        correct_bat_degree = 0

    supervised_tokens = int(
        (
            training_output
            .sequence_batch
            .labels
            != -100
        )
        .sum()
        .item()
    )

    diagnostics = (
        training_output.diagnostics
    )

    peak_cuda_memory = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
        if torch.cuda.is_available()
        else 0.0
    )

    print("\nObject Graph v1 training summary:")
    print(
        "  Loss:",
        f"{training_output.loss.item():.8f}",
    )
    print(
        "  Supervised tokens:",
        supervised_tokens,
    )
    print(
        "  First Pass extraction calls:",
        extraction_calls["count"],
    )
    print(
        "  Nodes:",
        len(nodes),
    )
    print(
        "  Source counts:",
        dict(source_counts),
    )
    print(
        "  Class counts:",
        dict(class_counts),
    )
    print(
        "  Edges:",
        len(edges),
    )
    print(
        "  Correct bat fused:",
        correct_bat_fused,
    )
    print(
        "  Correct bat degree:",
        correct_bat_degree,
    )

    print("\nGradient norms:")
    print(
        "  Global:",
        f"{global_gradient:.8f}",
    )
    print(
        "  Object Graph v1:",
        f"{graph_gradient:.8f}",
    )
    print(
        "  QLoRA:",
        f"{adapter_gradient:.8f}",
    )
    print(
        "  Clipped total:",
        f"{clipped_gradient:.8f}",
    )

    print("\nParameter deltas:")
    print(
        "  Global:",
        f"{global_delta:.8f}",
    )
    print(
        "  Object Graph v1:",
        f"{graph_delta:.8f}",
    )
    print(
        "  QLoRA:",
        f"{adapter_delta:.8f}",
    )

    print("\nIsolation diagnostics:")
    print(
        "  Base LLM has gradient:",
        base_llm_has_gradient,
    )
    print(
        "  Vision/mlp1 has gradient:",
        vision_or_mlp1_has_gradient,
    )
    print(
        "  Input order:",
        diagnostics.get("input_order"),
    )
    print(
        "  Graph version:",
        diagnostics.get("graph_version"),
    )
    print(
        "  Detection count:",
        diagnostics.get(
            "detection_count"
        ),
    )
    print(
        "  Peak CUDA memory:",
        f"{peak_cuda_memory:.2f} GiB",
    )

    checks = {
        "graph_version": (
            integrated_model.graph_version
            == "v1"
        ),
        "trainable_parameter_count": (
            total_trainable_count
            == 9_007_098
        ),
        "statistics_fitted": (
            integrated_model
            .descriptor_statistics_fitted
        ),
        "first_pass_not_recomputed": (
            extraction_calls["count"] == 0
        ),
        "first_pass_reused": (
            training_output.first_pass
            is first_pass
        ),
        "finite_loss": bool(
            torch.isfinite(
                training_output.loss
            )
        ),
        "positive_loss": (
            training_output.loss.item()
            > 0
        ),
        "answer_supervision": (
            supervised_tokens > 0
        ),
        "global_token_shape": (
            tuple(
                training_output
                .global_output
                .global_token
                .shape
            )
            == (1, 1, 3584)
        ),
        "graph_token_shape": (
            tuple(
                graph_output
                .graph_token
                .shape
            )
            == (1, 1, 3584)
        ),
        "correct_bat_fused": (
            correct_bat_fused
        ),
        "correct_bat_connected": (
            correct_bat_degree > 0
        ),
        "graph_nonempty": (
            len(edges) > 0
        ),
        "global_gradient": (
            global_gradient > 0
        ),
        "graph_gradient": (
            graph_gradient > 0
        ),
        "qlora_gradient": (
            adapter_gradient > 0
        ),
        "global_updated": (
            global_delta > 0
        ),
        "graph_updated": (
            graph_delta > 0
        ),
        "qlora_updated": (
            adapter_delta > 0
        ),
        "base_llm_no_gradient": (
            not base_llm_has_gradient
        ),
        "vision_and_mlp1_no_gradient": (
            not vision_or_mlp1_has_gradient
        ),
        "input_order": (
            diagnostics.get("input_order")
            == (
                "question_global_graph_"
                "image_answer"
            )
        ),
        "diagnostic_graph_version": (
            diagnostics.get(
                "graph_version"
            )
            == "v1"
        ),
        "diagnostic_detection_count": (
            diagnostics.get(
                "detection_count"
            )
            == len(detections)
        ),
    }

    print("\nChecks:")

    for name, passed in checks.items():
        print(
            f"  {name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    failed = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    if failed:
        raise RuntimeError(
            "Object Graph v1 training "
            f"test failed: {failed}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "graph_version": "v1",
            "question": QUESTION,
            "answer": ANSWER,
            "loss": float(
                training_output
                .loss
                .detach()
                .item()
            ),
            "trainable_parameters": (
                total_trainable_count
            ),
            "global_gradient": (
                global_gradient
            ),
            "graph_gradient": (
                graph_gradient
            ),
            "adapter_gradient": (
                adapter_gradient
            ),
            "global_delta": global_delta,
            "graph_delta": graph_delta,
            "adapter_delta": adapter_delta,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "correct_bat_fused": (
                correct_bat_fused
            ),
            "correct_bat_degree": (
                correct_bat_degree
            ),
            "diagnostics": diagnostics,
            "smoke_test": True,
        },
        OUTPUT_PATH,
    )

    print(
        "\nSaved artifact:",
        OUTPUT_PATH,
    )

    print(
        "\nObject Graph v1 training "
        "test: PASS"
    )
    print(
        "Note: this validates one optimizer "
        "step and gradient isolation, not "
        "task performance."
    )


if __name__ == "__main__":
    main()
