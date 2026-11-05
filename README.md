# Global-Relational Token Fusion for Two-Pass VLM Inference

> Research prototype for question-conditioned Global and Graph token fusion with InternVL3, NF4 quantization, and QLoRA.

## Status

This repository contains an experimental implementation accompanying an unpublished research project. The architecture and training pipeline are implemented end to end, but large-scale baselines, ablations, multi-seed experiments, and paper-level performance validation are still in progress.

The reported Debug-100 values are pipeline verification results, not benchmark or paper performance claims.

## Overview

The model performs two vision-language inference passes.

1. A fixed first pass extracts question-conditioned visual attention and projected visual features.
2. The attention information is encoded into one Global Token and one Graph Token.
3. A second language-model pass receives the reordered sequence:

```text
Question → Global Token → Graph Token → Image Tokens
```

Only the second-pass language model uses the trainable QLoRA adapter. The first pass keeps the adapter disabled so that attention maps remain fixed and can be cached.

## Architecture

```text
Image + Question
       │
       ▼
InternVL3 First Pass
(Adapter OFF, eager attention)
       │
       ├── Local attention map
       ├── Layer/head attention profiles
       └── Projected visual features
              │
       ┌──────┴──────────────────┐
       │                         │
       ▼                         ▼
Global Token Branch       Graph Token Branch
       │                         │
Spatial CNN               Top-K salient regions
       +                  Weighted relational graph
16 descriptors            Two-layer weighted GNN
       +                  Mean readout
AGI-MLP                          │
       │                         │
Conditional FiLM                │
+ gated interaction             │
       │                         │
       ▼                         ▼
One Global Token          One Graph Token
       └──────────┬──────────────┘
                  ▼
InternVL3 Second Pass
(QLoRA Adapter ON)
```

### Global Token branch

The question-conditioned attention distribution is represented through two complementary branches.

- Spatial branch: a lightweight CNN processes a normalized 32×32 attention map.
- Statistical branch: an Adaptive Group Interaction MLP processes 16 descriptors.
- Fusion: descriptor-conditioned FiLM, feature-wise gating, and multiplicative interaction.
- Projection: the fused 128-dimensional representation is projected to one 3,584-dimensional Global Token.

The descriptor groups are:

- Concentration: peak value, top-5% mass, spatial entropy, Hoyer sparsity
- Spatial bias: centroid x/y, left-right balance, top-bottom balance
- Shape: spread x/y, eccentricity, peak count
- Frequency: low/mid/high-frequency energy, spectral entropy

### Graph Token branch

- Selects the top 16 salient local image tokens
- Builds node features from projected visual features, attention weights, and positions
- Combines spatial, visual, and layer/head attention similarities
- Applies a two-layer weighted GNN
- Uses mean readout and projects the graph representation to one 3,584-dimensional Graph Token

### Pass isolation

```text
First Pass:  Base InternVL3, Adapter OFF
Second Pass: NF4 InternVL3 + trainable QLoRA Adapter
```

This separation keeps cached first-pass attention stable while allowing the second-pass language model to adapt to the auxiliary tokens.

## Repository structure

```text
configs/    Descriptor normalization statistics
data/       GQA Dataset and DataLoader utilities
models/     Global, Graph, fusion, injection, and integrated model modules
scripts/    Data preparation, tests, caching, training, and evaluation
utils/      Statistics, cache, checkpoint, and training runtime utilities
```

Generated outputs, datasets, pretrained weights, checkpoints, caches, and personal test images are intentionally excluded from Git.

## Tested environment

- Python 3.10.22
- PyTorch 2.5.1 + CUDA 12.1
- torchvision 0.20.1
- transformers 4.48.3
- accelerate 1.2.1
- PEFT 0.14.0
- bitsandbytes 0.45.1
- NVIDIA GeForce RTX 4070 Ti

Eager attention is required for selective layer/head attention extraction. FlashAttention2 is not used in the current first-pass implementation.

## Installation

The current path layout expects this repository to be cloned into a directory named `vlm`.

```bash
mkdir global-relational-workspace
cd global-relational-workspace

git clone \
git@github.com:jyj1256/global-relational-vlm.git \
vlm

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r vlm/requirements.txt
```

The CUDA-enabled PyTorch packages in `requirements.txt` target CUDA 12.1. Adjust them when using a different CUDA environment.

## External assets

The repository does not redistribute InternVL3 weights or the GQA dataset.

Expected workspace layout:

```text
global-relational-workspace/
├── vlm/
├── datasets/
│   └── gqa/
│       ├── images/
│       ├── questions/
│       ├── sceneGraphs/
│       └── subsets/
└── pretrained/
    ├── huggingface/
    └── offload/
```

- Backbone: [OpenGVLab/InternVL3-8B](https://huggingface.co/OpenGVLab/InternVL3-8B)
- Dataset: [GQA: Visual Reasoning in the Real World](https://cs.stanford.edu/people/dorarad/gqa/index.html)

Users are responsible for complying with the model and dataset licenses.

## Core workflow

Run commands from the parent workspace directory, not from inside `vlm`.

### 1. Validate GQA

```bash
python vlm/scripts/validate_gqa_data.py
python vlm/scripts/create_gqa_subsets.py
python vlm/scripts/create_gqa_training_subsets.py
```

### 2. Compute descriptor statistics

```bash
HF_HUB_OFFLINE=1 \
python vlm/scripts/compute_descriptor_statistics.py \
--first-pass nf4-qlora \
--sample-count 5000
```

The final NF4 statistics used by the current training pipeline are included in:

```text
vlm/configs/gqa_descriptor_statistics_nf4_qlora_5000_seed2026.json
```

### 3. Cache fixed First Passes

```bash
HF_HUB_OFFLINE=1 \
python vlm/scripts/cache_gqa_first_passes.py \
--manifest \
datasets/gqa/subsets/train_short_1000_seed2026.jsonl \
--progress-every 50
```

### 4. Train

```bash
HF_HUB_OFFLINE=1 \
python vlm/scripts/train_global_relational.py \
--output-dir \
vlm/outputs/training/debug100_full_epoch1 \
--checkpoint-every 5
```

### 5. Evaluate

```bash
HF_HUB_OFFLINE=1 \
python vlm/scripts/evaluate_gqa.py \
--progress-every 10
```

## Pipeline verification results

These values only verify that the implementation trains, resumes, and evaluates correctly.

| Check | Result |
|---|---:|
| Trainable auxiliary + QLoRA parameters | 7,844,251 |
| Exact checkpoint Resume error | 0 across 23,538,170 tensor values |
| Debug Train samples | 100 |
| Debug Validation samples | 100 |
| Debug Validation strict exact match | 52/100 |
| Peak training CUDA memory | 9.46 GiB |
| Cached generation time | 0.427 s/sample |

The cached generation time excludes first-pass computation and model-loading time. The validation subset is structurally balanced and is not representative of the full official GQA distribution.

## Reproducibility notes

- Default seed: 2026
- First-pass attention layers: zero-based layers 10–17
- Attention reference: last non-special question token
- Local attention