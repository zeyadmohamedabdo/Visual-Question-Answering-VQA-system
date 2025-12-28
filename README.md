# Visual Question Answering (VQA) System

> **Advanced CNN Project** - Deep Learning Course  
> A complete VQA system using custom CNN backbone, Transformer text encoder, and cross-attention fusion.

---

## Table of Contents

1. [Problem Definition](#problem-definition)
2. [Dataset & Subset Justification](#dataset--subset-justification)
3. [Architecture](#architecture)
4. [Attention Mechanisms](#attention-mechanisms)
5. [Training Setup](#training-setup)
6. [How to Run](#how-to-run)
7. [Project Structure](#project-structure)
8. [Ablation Study](#ablation-study)
9. [Team Distribution](#team-distribution)
10. [References](#references)

---

## Problem Definition

**Visual Question Answering (VQA)** is a multimodal task that requires understanding both:
- **Visual content** from an image
- **Natural language** from a question

The goal is to predict a correct answer given an image-question pair.

### Task Formulation

We formulate VQA as a **multi-class classification** problem:

```
Input:  Image I ∈ ℝ^(3×224×224), Question Q = {w₁, w₂, ..., wₙ}
Output: Answer a ∈ {a₁, a₂, ..., a₁₀₀₀} (top-1000 most frequent answers)
```

**Why Classification (not Generation)?**
1. Most VQA questions have short, common answers ("yes", "no", "blue", "2")
2. Standard cross-entropy loss enables stable training
3. Evaluation is straightforward (accuracy)
4. Follows the VQA v2 challenge setup

---

## Dataset & Subset Justification

### Datasets Used

| Dataset | Purpose | Size |
|---------|---------|------|
| **MS-COCO val2017** | Images | ~5,000 images |
| **VQA v2.0 val** | Questions & Answers | ~214,000 Q&A pairs |

### Subset Selection

Due to computational constraints (academic project), we use:

- **25,000 Q&A pairs** (approximately 12% of validation set)
- **80/20 train/validation split** (20,000 train, 5,000 val)
- **Top-1000 most frequent answers** (covers ~87% of all answers)

**Justification:**
1. Training CNN from scratch requires significant GPU memory
2. VQA validation set is representative of overall distribution
3. Top-1000 answers provide sufficient coverage for meaningful evaluation
4. Subset allows complete training within reasonable time (~2-4 hours on GPU)

---

## Architecture

### Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                       VQA Model Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Image [3×224×224]        Question "What color is the car?"    │
│         │                              │                         │
│         ▼                              ▼                         │
│   ┌───────────────┐            ┌───────────────┐                │
│   │ Custom ResNet │            │  Transformer  │                │
│   │ + SE + Spatial│            │    Encoder    │                │
│   │   Attention   │            │ (4 layers)    │                │
│   └───────────────┘            └───────────────┘                │
│         │                              │                         │
│   [512×7×7]                    [seq_len×256]                    │
│         │                              │                         │
│         └──────────┬───────────────────┘                        │
│                    ▼                                             │
│           ┌───────────────┐                                      │
│           │Cross-Attention│                                      │
│           │ Q=text, K/V=img│                                     │
│           └───────────────┘                                      │
│                    │                                             │
│                 [256]                                            │
│                    ▼                                             │
│           ┌───────────────┐                                      │
│           │  Answer Head  │                                      │
│           │   MLP→1000    │                                      │
│           └───────────────┘                                      │
│                    │                                             │
│             Answer logits                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Component Details

#### 1. Image Encoder (Custom ResNet)

**NO PRETRAINED WEIGHTS** - trained from scratch.

| Stage | Output Shape | Details |
|-------|--------------|---------|
| Stem | [64, 56, 56] | Conv7×7/2 → BN → ReLU → MaxPool |
| Stage 1 | [64, 56, 56] | 2× ResBlock + SE Attention |
| Stage 2 | [128, 28, 28] | 2× ResBlock + SE Attention |
| Stage 3 | [256, 14, 14] | 2× ResBlock + SE + Spatial |
| Stage 4 | [512, 7, 7] | 2× ResBlock + SE + Spatial |

**Key Design Choice:** Output is a **feature map** (not a vector), preserving spatial information for cross-attention.

#### 2. Text Encoder (Transformer)

| Component | Specification |
|-----------|---------------|
| Embedding Dim | 256 |
| Layers | 4 |
| Attention Heads | 8 |
| FFN Hidden | 1024 |
| Positional Encoding | Sinusoidal |

**Implementation:** Fully custom multi-head self-attention (NOT using `nn.MultiheadAttention`).

#### 3. Cross-Attention Fusion

Question features **query** image features:

```
Q = W_q × TextFeatures    [B, L_text, 256]
K = W_k × ImageFeatures   [B, 49, 256]
V = W_v × ImageFeatures   [B, 49, 256]

Attention = softmax(QK^T / √d) × V
```

This allows the model to attend to relevant image regions based on the question.

#### 4. Answer Head

Three-layer MLP with dropout:
```
[256] → ReLU → Dropout → [512] → ReLU → Dropout → [1000]
```

---

## Attention Mechanisms

### SE (Squeeze-and-Excitation) Attention

**Purpose:** Channel-wise recalibration

**Mathematical Formulation:**

Given input X ∈ ℝ^(C×H×W):

1. **Squeeze:** z_c = GlobalAvgPool(X_c) → z ∈ ℝ^C
2. **Excitation:** s = σ(W₂ · ReLU(W₁ · z)) → s ∈ ℝ^C
3. **Scale:** X' = s ⊙ X

**Intuition:** Different channels encode different features (edges, textures, objects). SE learns which channels are important for the current input.

### Spatial Attention

**Purpose:** Location-based weighting

**Mathematical Formulation:**

Given input X ∈ ℝ^(C×H×W):

1. **Pool:** F_max = max_c(X), F_avg = mean_c(X)
2. **Concat:** F = [F_max; F_avg] ∈ ℝ^(2×H×W)
3. **Conv:** M = σ(Conv7×7(F)) ∈ ℝ^(1×H×W)
4. **Scale:** X' = M ⊙ X

**Intuition:** For VQA, different image regions have different relevance. Spatial attention highlights important areas.

### Cross-Attention

**Purpose:** Question-conditioned visual attention

**Why Question as Query?**
- Question defines "what to look for"
- Image provides "where to look"
- Enables question-specific visual grounding

---

## Training Setup

### Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Learning Rate | 1e-4 | Standard for training from scratch |
| Optimizer | AdamW | Better weight decay than Adam |
| Weight Decay | 0.01 | Regularization |
| Scheduler | CosineAnnealing | Smooth LR decay |
| Batch Size | 32 | Memory efficient |
| Epochs | 30 | Sufficient for convergence |
| Gradient Clip | 1.0 | Stability |

### Loss Function

**CrossEntropyLoss** - standard for multi-class classification:

```
L = -log(softmax(logits)[target])
```

### Training Command

```bash
# Full training
python training/train.py --epochs 30 --batch-size 32

# Quick test
python training/train.py --demo --epochs 1 --subset 100

# Ablation (no attention)
python training/train.py --no-attention
```

---

## How to Run

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ GPU memory

### Installation

```bash
# Clone/navigate to project
cd d:/cnn

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

1. Download MS-COCO val2017:
   - Images: http://images.cocodataset.org/zips/val2017.zip
   - Extract to: `d:/cnn/data/raw/val2017/`

2. Download VQA v2:
   - https://visualqa.org/download.html
   - Extract to: `d:/cnn/data/raw/vqa/`

### Training

```bash
# With real data
python training/train.py

# With demo data (for testing)
python training/train.py --demo
```

### Evaluation

```bash
python training/evaluate.py --checkpoint checkpoints/best_model.pth
```

### Running the Demo

```bash
# Start API server
cd d:/cnn
uvicorn api.main:app --reload --port 8000

# Open frontend
# Navigate to: d:/cnn/frontend/index.html in browser
```

---

## Project Structure

```
VQA-Project/
│
├── models/                    # Neural network modules
│   ├── attention_modules.py   # SE, Spatial, CBAM attention
│   ├── cnn_backbone.py        # Custom ResNet
│   ├── text_encoder.py        # Transformer encoder
│   ├── cross_attention.py     # Cross-attention fusion
│   ├── fusion.py              # Multimodal fusion
│   └── vqa_model.py           # Complete model
│
├── data/                      # Data processing
│   ├── build_vocab.py         # Answer vocabulary
│   ├── preprocess.py          # Image/text preprocessing
│   └── dataset.py             # PyTorch Dataset
│
├── utils/                     # Utilities
│   ├── tokenizer.py           # Word tokenization
│   ├── metrics.py             # Accuracy metrics
│   └── config.py              # Configuration
│
├── training/                  # Training pipeline
│   ├── train.py               # Training loop
│   └── evaluate.py            # Evaluation
│
├── api/                       # Backend API
│   ├── inference.py           # Inference utilities
│   └── main.py                # FastAPI endpoints
│
├── frontend/                  # Web GUI
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── README.md
└── requirements.txt
```

---

## Ablation Study

### Experiment: Effect of Attention Mechanisms

| Model | Top-1 Acc | Top-5 Acc | Parameters |
|-------|-----------|-----------|------------|
| Full (SE + Spatial) | - | - | ~25M |
| No Spatial | - | - | ~24M |
| No Attention | - | - | ~23M |

**To run ablation:**
```bash
# Full model
python training/train.py

# No attention
python training/train.py --no-attention
```

---

## Team Distribution

| Member | Responsibilities |
|--------|-----------------|
| **Zeyad Mohamed** | CNN Backbone, SE/Spatial Attention |
| **Youssef Mahmoud** | Transformer Text Encoder |
| **Mariam aboalhasan** | Cross-Attention, Fusion Module |
| **Mokhtar Mohamed** | Training Pipeline, Metrics |
| **Youssef Fahmy** | API Backend, Frontend GUI |

---

## References

1. Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. CVPR.
2. Woo, S., et al. (2018). CBAM: Convolutional block attention module. ECCV.
3. Vaswani, A., et al. (2017). Attention is all you need. NeurIPS.
4. Antol, S., et al. (2015). VQA: Visual question answering. ICCV.
5. Anderson, P., et al. (2018). Bottom-up and top-down attention for VQA. CVPR.

---

## License

This project is for educational purposes as part of an Advanced CNN course.

---

*Last updated: December 2025*
