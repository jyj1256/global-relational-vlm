import sys
from collections import Counter
from pathlib import Path

import bitsandbytes as bnb
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

VLM_ROOT = PROJECT_ROOT / "vlm"

for import_path in (
    PROJECT_ROOT,
    VLM_ROOT,
):
    if str(import_path) not in sys.path:
        sys.path.insert(
            0,
            str(import_path),
        )


from scripts.test_backbone import (
    get_vision_device,
    load_image,
)
from vlm.models.global_relational_vlm import (
    GlobalRelationalVLM,
)
from vlm.models.object_graph_v1.encoder import (
    ObjectGraphTokenEncoderV1Output,
)
from vlm.models.qlora_adapter import (
    attach_qlora_adapter,
    has_qlora_adapter,
)
from vlm.utils.descriptor_statistics import (
    load_descriptor_statistics,
)
from vlm.utils.object_detection_cache import (
    load_object_detection_cache,
)


MODEL_NAME = "OpenGVLab/InternVL3-8B"

MODEL_CACHE_DIR = (
    PROJECT_ROOT
    / "pretrained/huggingface"
)

OFFLOAD_DIR = (
    PROJECT_ROOT
    / "pretrained/offload"
)

IMAGE_PATH = (
    VLM_ROOT
    / "outputs/paper_figures/"
    "baseball_2337002/input.jpg"
)

DETECTION_PATH = (
    VLM_ROOT
    / "outputs/object_detections/"
    "yolo11s_coco/entries/2337002.json"
)

STATISTICS_PATH = (
    VLM_ROOT
    / "configs/"
    "gqa_descriptor_statistics_"
    "nf4_qlora_5000_seed2026.json"
)

OUTPUT_PATH = (
    VLM_ROOT
    / "outputs/predictions/"
    "untrained_global_relational_v1_"
    "baseball_test.pt"
)

QUESTION = (
    "What is the man to the left "
    "of the people holding?"
)

REFERENCE_ANSWER = "bat"


def main():
    required_paths = (
        IMAGE_PATH,
        DETECTION_PATH,
        STATISTICS_PATH,
    )

    for required_path in required_paths:
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

    print("Global-Relational v1 sample:")
    print("  Image:", IMAGE_PATH)
    print("  Question:", QUESTION)
    print("  Reference answer:", REFERENCE_ANSWER)
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
        cache_dir=str(MODEL_CACHE_DIR),
        use_fast=False,
        local_files_only=True,
    )

    print("Loading NF4 InternVL3-8B...")

    backbone = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        cache_dir=str(MODEL_CACHE_DIR),
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

    adapter_report = attach_qlora_adapter(
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

    statistics = load_descriptor_statistics(
        input_path=STATISTICS_PATH,
    )

    integrated_model.set_descriptor_statistics(
        mean=statistics["mean"],
        std=statistics["std"],
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
        raise ValueError(
            "Detection image size does not "
            "match the input image."
        )

    vision_device = get_vision_device(
        backbone
    )

    pixel_values = pixel_values.to(
        device=vision_device,
        dtype=torch.bfloat16,
    )

    integrated_model.global_encoder.to(
        vision_device
    )
    integrated_model.graph_encoder.to(
        vision_device
    )

    integrated_model.freeze_backbone()
    integrated_model.eval()

    quantized_module_names = [
        name
        for name, module
        in backbone.named_modules()
        if isinstance(
            module,
            bnb.nn.Linear4bit,
        )
    ]

    outside_language_quantized = [
        name
        for name in quantized_module_names
        if not name.startswith(
            "language_model"
        )
    ]

    print("\nIntegrated v1 model:")
    print(
        "  Graph version:",
        integrated_model.graph_version,
    )
    print(
        "  Auxiliary parameters:",
        f"{integrated_model.auxiliary_parameter_count():,}",
    )
    print(
        "  QLoRA parameters:",
        f"{adapter_report.trainable_adapter_parameters:,}",
    )
    print(
        "  Descriptor statistics fitted:",
        integrated_model
        .descriptor_statistics_fitted,
    )
    print(
        "  Pixel tensor:",
        tuple(pixel_values.shape),
    )
    print("  Tile grid:", tile_grid)
    print("  Vision device:", vision_device)

    print(
        "\nRunning final Object Graph v1 "
        "Two-Pass inference..."
    )

    generation_output = (
        integrated_model.generate(
            pixel_values=pixel_values,
            question=QUESTION,
            tile_grid=tile_grid,
            image_size=original_image.size,
            generation_config={
                "max_new_tokens": 32,
                "do_sample": False,
                "num_beams": 1,
            },
            detections=detections,
        )
    )

    graph_output = (
        generation_output.graph_output
    )

    if not isinstance(
        graph_output,
        ObjectGraphTokenEncoderV1Output,
    ):
        raise TypeError(
            "Expected Object Graph v1 output."
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
        correct_bat_node = (
            correct_bat_nodes[0]
        )

        correct_bat_degree = int(
            graph_output
            .graph_output
            .degree[
                correct_bat_node.node_id
            ]
            .item()
        )

        correct_bat_weight = float(
            graph_output
            .readout_output
            .node_weights[
                correct_bat_node.node_id
            ]
            .item()
        )
    else:
        correct_bat_degree = 0
        correct_bat_weight = 0.0

    diagnostics = (
        generation_output
        .second_pass_diagnostics
    )

    complete_edge_count = (
        len(nodes)
        * (len(nodes) - 1)
        // 2
    )

    global_token = (
        generation_output
        .global_output
        .global_token
    )

    graph_token = graph_output.graph_token

    print("\nFirst Pass:")
    print(
        "  Attention:",
        tuple(
            generation_output
            .first_pass
            .attention_result
            .local_image_l1.shape
        ),
    )
    print(
        "  Attention sum:",
        float(
            generation_output
            .first_pass
            .attention_result
            .local_image_l1
            .sum()
            .item()
        ),
    )
    print(
        "  Profiles:",
        tuple(
            generation_output
            .first_pass
            .local_attention_profiles
            .shape
        ),
    )
    print(
        "  Visual features:",
        tuple(
            generation_output
            .first_pass
            .projected_visual_features
            .shape
        ),
    )

    print("\nObject Graph v1:")
    print("  Nodes:", len(nodes))
    print("  Source counts:", dict(source_counts))
    print("  Class counts:", dict(class_counts))
    print(
        "  Undirected edges:",
        graph_output
        .graph_output
        .undirected_edge_count,
    )
    print(
        "  Correct bat fused:",
        correct_bat_fused,
    )
    print(
        "  Correct bat degree:",
        correct_bat_degree,
    )
    print(
        "  Correct bat readout weight:",
        f"{correct_bat_weight:.8f}",
    )

    print("\nAuxiliary Tokens:")
    print(
        "  Global Token:",
        tuple(global_token.shape),
    )
    print(
        "  Global norm:",
        float(
            global_token.float().norm().item()
        ),
    )
    print(
        "  Graph Token:",
        tuple(graph_token.shape),
    )
    print(
        "  Graph norm:",
        float(
            graph_token.float().norm().item()
        ),
    )

    print("\nSecond Pass:")
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
        "  QLoRA attached:",
        diagnostics.get("qlora_attached"),
    )
    print(
        "  Adapter enabled:",
        diagnostics.get("adapter_enabled"),
    )

    print("\nGenerated response:")
    print(generation_output.response)

    checks = {
        "graph_version": (
            integrated_model.graph_version
            == "v1"
        ),
        "auxiliary_parameter_count": (
            integrated_model
            .auxiliary_parameter_count()
            == 3_960_826
        ),
        "statistics_fitted": (
            integrated_model
            .descriptor_statistics_fitted
        ),
        "detections_loaded": (
            len(detections) == 30
        ),
        "person_detection": (
            class_counts["person"] > 0
        ),
        "bat_detection": (
            class_counts["baseball bat"] > 0
        ),
        "glove_detection": (
            class_counts["baseball glove"] > 0
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
        "graph_sparse": (
            len(edges)
            < complete_edge_count
        ),
        "global_token_shape": (
            tuple(global_token.shape)
            == (1, 1, 3584)
        ),
        "graph_token_shape": (
            tuple(graph_token.shape)
            == (1, 1, 3584)
        ),
        "finite_global_token": bool(
            torch.isfinite(
                global_token
            ).all()
        ),
        "finite_graph_token": bool(
            torch.isfinite(
                graph_token
            ).all()
        ),
        "node_weights_normalized": bool(
            torch.allclose(
                graph_output
                .readout_output
                .node_weights
                .sum(),
                torch.ones(
                    (),
                    device=(
                        graph_output
                        .readout_output
                        .node_weights
                        .device
                    ),
                ),
                atol=1e-5,
                rtol=1e-5,
            )
        ),
        "language_quantized": (
            len(quantized_module_names) > 0
        ),
        "only_language_quantized": (
            len(
                outside_language_quantized
            )
            == 0
        ),
        "qlora_attached": (
            has_qlora_adapter(backbone)
        ),
        "input_order": (
            diagnostics.get("input_order")
            == "question_global_graph_image"
        ),
        "diagnostic_graph_version": (
            diagnostics.get("graph_version")
            == "v1"
        ),
        "diagnostic_detection_count": (
            diagnostics.get(
                "detection_count"
            )
            == len(detections)
        ),
        "second_pass_adapter": (
            diagnostics.get(
                "adapter_enabled"
            )
            is True
        ),
        "response_nonempty": bool(
            generation_output
            .response
            .strip()
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
            "Global-Relational v1 pipeline "
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
            "reference_answer": (
                REFERENCE_ANSWER
            ),
            "response": (
                generation_output.response
            ),
            "detection_count": len(
                detections
            ),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "source_counts": dict(
                source_counts
            ),
            "class_counts": dict(
                class_counts
            ),
            "correct_bat_fused": (
                correct_bat_fused
            ),
            "correct_bat_degree": (
                correct_bat_degree
            ),
            "correct_bat_weight": (
                correct_bat_weight
            ),
            "global_token": (
                global_token
                .detach()
                .cpu()
            ),
            "graph_token": (
                graph_token
                .detach()
                .cpu()
            ),
            "second_pass_diagnostics": (
                diagnostics
            ),
            "trained": False,
        },
        OUTPUT_PATH,
    )

    if torch.cuda.is_available():
        peak_memory = (
            torch.cuda
            .max_memory_allocated()
            / (1024 ** 3)
        )

        print(
            "\nPeak CUDA memory:",
            f"{peak_memory:.2f} GiB",
        )

    print(
        "\nSaved artifact:",
        OUTPUT_PATH,
    )

    print(
        "\nGlobal-Relational Object Graph "
        "v1 pipeline test: PASS"
    )
    print(
        "Note: this validates the complete "
        "v1 inference path, not task "
        "performance. The v1 auxiliary "
        "encoders are untrained."
    )


if __name__ == "__main__":
    main()
