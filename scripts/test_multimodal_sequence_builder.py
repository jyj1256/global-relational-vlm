import sys
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from vlm.models.multimodal_sequence_builder import (
    build_training_sequence,
)


class FakeTokenizer:
    def __init__(self):
        self.token_ids = {
            "<img>": 100,
            "</img>": 101,
            "<|im_end|>": 102,
        }

    def convert_tokens_to_ids(self, token):
        return self.token_ids[token]


def main():
    torch.manual_seed(2026)

    tokenizer = FakeTokenizer()

    hidden_dimension = 8

    # Original prompt layout:
    #
    # prefix:
    #   [1, 2, 3]
    #
    # image:
    #   [<img>, image_1, image_2, </img>]
    #
    # question:
    #   [10, 11, 12]
    #
    # suffix:
    #   [<|im_end|>, assistant, newline]
    prompt_input_ids = torch.tensor(
        [
            [
                1,
                2,
                3,
                100,
                200,
                201,
                101,
                10,
                11,
                12,
                102,
                13,
                14,
            ]
        ],
        dtype=torch.long,
    )

    prompt_length = prompt_input_ids.shape[1]

    prompt_input_embeddings = torch.arange(
        prompt_length * hidden_dimension,
        dtype=torch.float32,
    ).view(
        1,
        prompt_length,
        hidden_dimension,
    )

    prompt_attention_mask = torch.ones(
        (1, prompt_length),
        dtype=torch.long,
    )

    global_token = torch.full(
        (1, 1, hidden_dimension),
        fill_value=-1.0,
        requires_grad=True,
    )

    graph_token = torch.full(
        (1, 1, hidden_dimension),
        fill_value=-2.0,
        requires_grad=True,
    )

    # answer:
    #   [owl_token_1, owl_token_2, <|im_end|>]
    answer_input_ids = torch.tensor(
        [[20, 21, 102]],
        dtype=torch.long,
    )

    answer_embeddings = torch.full(
        (1, 3, hidden_dimension),
        fill_value=3.0,
    )

    output = build_training_sequence(
        tokenizer=tokenizer,
        prompt_input_ids=prompt_input_ids,
        prompt_input_embeddings=(
            prompt_input_embeddings
        ),
        prompt_attention_mask=(
            prompt_attention_mask
        ),
        global_token=global_token,
        graph_token=graph_token,
        answer_input_ids=answer_input_ids,
        answer_embeddings=answer_embeddings,
    )

    diagnostics = output.diagnostics

    print("Training Sequence summary:")

    for key, value in diagnostics.items():
        print(f"  {key}: {value}")

    print("\nTensor shapes:")
    print(
        "  inputs_embeds:",
        tuple(output.inputs_embeds.shape),
    )
    print(
        "  attention_mask:",
        tuple(output.attention_mask.shape),
    )
    print(
        "  labels:",
        tuple(output.labels.shape),
    )

    global_index = diagnostics[
        "global_token_index"
    ]

    graph_index = diagnostics[
        "graph_token_index"
    ]

    image_start = diagnostics[
        "reordered_image_start"
    ]

    answer_start = diagnostics[
        "answer_start"
    ]

    print("\nImportant positions:")
    print("  Global index:", global_index)
    print("  Graph index:", graph_index)
    print("  Image start:", image_start)
    print("  Answer start:", answer_start)
    print(
        "  Global label:",
        output.labels[0, global_index].item(),
    )
    print(
        "  Graph label:",
        output.labels[0, graph_index].item(),
    )
    print(
        "  Answer labels:",
        output.labels[0, answer_start:].tolist(),
    )

    # Verify preserved embedding segments.
    expected_prefix = (
        prompt_input_embeddings[:, 0:3]
    )

    expected_question = (
        prompt_input_embeddings[:, 7:10]
    )

    expected_image = (
        prompt_input_embeddings[:, 3:7]
    )

    expected_suffix = (
        prompt_input_embeddings[:, 10:13]
    )

    prefix_matches = torch.allclose(
        output.inputs_embeds[:, 0:3],
        expected_prefix,
    )

    question_matches = torch.allclose(
        output.inputs_embeds[:, 3:6],
        expected_question,
    )

    image_matches = torch.allclose(
        output.inputs_embeds[:, 8:12],
        expected_image,
    )

    suffix_matches = torch.allclose(
        output.inputs_embeds[:, 12:15],
        expected_suffix,
    )

    answer_matches = torch.allclose(
        output.inputs_embeds[:, 15:18],
        answer_embeddings,
    )

    prompt_labels = output.labels[
        :,
        :answer_start,
    ]

    answer_labels = output.labels[
        :,
        answer_start:,
    ]

    # Verify that a standard causal CE loss can be
    # calculated using the constructed labels.
    vocabulary_size = 256

    fake_logits = torch.randn(
        1,
        output.labels.shape[1],
        vocabulary_size,
        requires_grad=True,
    )

    shifted_logits = fake_logits[:, :-1]
    shifted_labels = output.labels[:, 1:]

    test_loss = F.cross_entropy(
        shifted_logits.reshape(
            -1,
            vocabulary_size,
        ),
        shifted_labels.reshape(-1),
        ignore_index=-100,
    )

    test_loss.backward()

    # Verify gradient paths through both inserted tokens.
    auxiliary_loss = (
        output.inputs_embeds[
            :,
            global_index,
            :,
        ].square().mean()
        + output.inputs_embeds[
            :,
            graph_index,
            :,
        ].square().mean()
    )

    auxiliary_loss.backward()

    global_gradient = float(
        global_token.grad.norm().item()
    )

    graph_gradient = float(
        graph_token.grad.norm().item()
    )

    print("\nLoss diagnostics:")
    print(
        "  Test Cross-Entropy loss:",
        f"{test_loss.item():.8f}",
    )
    print(
        "  Global token gradient:",
        f"{global_gradient:.8f}",
    )
    print(
        "  Graph token gradient:",
        f"{graph_gradient:.8f}",
    )

    checks = {
        "embedding_shape": (
            output.inputs_embeds.shape
            == (1, 18, hidden_dimension)
        ),
        "mask_shape": (
            output.attention_mask.shape
            == (1, 18)
        ),
        "label_shape": (
            output.labels.shape == (1, 18)
        ),
        "sequence_length": (
            diagnostics[
                "training_sequence_length"
            ] == prompt_length + 2 + 3
        ),
        "input_order": (
            diagnostics["input_order"]
            == "question_global_graph_image_answer"
        ),
        "global_index": global_index == 6,
        "graph_index": graph_index == 7,
        "image_start": image_start == 8,
        "answer_start": answer_start == 15,
        "prefix_preserved": prefix_matches,
        "question_preserved": question_matches,
        "image_preserved": image_matches,
        "suffix_preserved": suffix_matches,
        "answer_preserved": answer_matches,
        "prompt_labels_masked": bool(
            prompt_labels.eq(-100).all().item()
        ),
        "answer_labels": bool(
            torch.equal(
                answer_labels,
                answer_input_ids,
            )
        ),
        "global_label_masked": (
            output.labels[
                0,
                global_index,
            ].item()
            == -100
        ),
        "graph_label_masked": (
            output.labels[
                0,
                graph_index,
            ].item()
            == -100
        ),
        "supervised_count": (
            diagnostics[
                "supervised_token_count"
            ] == 3
        ),
        "finite_loss": bool(
            torch.isfinite(test_loss).item()
        ),
        "global_gradient": (
            global_gradient > 0.0
        ),
        "graph_gradient": (
            graph_gradient > 0.0
        ),
    }

    failed_checks = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    print("\nChecks:")

    for name, passed in checks.items():
        print(
            f"  {name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    if failed_checks:
        raise RuntimeError(
            "Sequence Builder test failed: "
            + ", ".join(failed_checks)
        )

    print("\nMultimodal Sequence Builder test: PASS")


if __name__ == "__main__":
    main()