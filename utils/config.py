"""
Configuration Module for VQA System
====================================
This module centralizes all hyperparameters, model configurations, and path settings.
Having a single configuration source ensures reproducibility and easy experimentation.

Why This Structure:
- Separation of concerns: Model configs, training configs, and paths are distinct
- Easy ablation: Toggle attention mechanisms with boolean flags
- Environment flexibility: Dataset paths are configurable for different machines
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PathConfig:
    """
    Dataset and output paths configuration.
    
    All paths are configurable to support different environments.
    Default assumes data is in d:/cnn/data/raw/ as approved in implementation plan.
    """
    # Base project directory
    project_root: str = "d:/cnn"
    
    # Raw data directories (downloaded datasets)
    raw_data_dir: str = "d:/cnn/data/raw"
    coco_images_dir: str = "d:/cnn/data/raw/coco_val2017/images"  # COCO val2017 images
    vqa_annotations_dir: str = "d:/cnn/data/raw/vqa_v2"   # VQA v2 annotations
    
    # Processed data directories
    processed_dir: str = "d:/cnn/data/processed"
    vocab_file: str = "d:/cnn/data/processed/answer_vocab.json"
    question_vocab_file: str = "d:/cnn/data/processed/question_vocab.json"
    
    # Model checkpoints
    checkpoint_dir: str = "d:/cnn/checkpoints"
    best_model_path: str = "d:/cnn/checkpoints/best_model.pth"
    
    # Logs
    log_dir: str = "d:/cnn/logs"
    
    def __post_init__(self):
        """Create necessary directories if they don't exist."""
        dirs_to_create = [
            self.processed_dir,
            self.checkpoint_dir,
            self.log_dir
        ]
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)


@dataclass
class ModelConfig:
    """
    Model architecture hyperparameters.
    
    Design Rationale:
    - embed_dim=256: Balance between expressiveness and computational cost
    - num_heads=8: Standard choice allowing diverse attention patterns
    - num_transformer_layers=4: Sufficient for short VQA questions
    - cnn_output_channels=512: Matches standard ResNet feature dimensions
    """
    
    # =========================================================================
    # Image Encoder (Custom CNN) Configuration
    # =========================================================================
    # Input image dimensions (standard ImageNet size for compatibility)
    image_size: int = 224
    
    # CNN channel progression through stages
    # Stage 1: 64, Stage 2: 128, Stage 3: 256, Stage 4: 512
    cnn_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    
    # Final feature map spatial size after CNN (224 -> 7 via strided convs + pooling)
    cnn_feature_size: int = 7  # Results in 7x7 = 49 spatial locations
    
    # Final CNN output channels (used for cross-attention)
    cnn_output_channels: int = 512
    
    # =========================================================================
    # Attention Module Configuration
    # =========================================================================
    # SE Attention reduction ratio (channel dimension reduced by this factor)
    # Lower ratio = more capacity but higher compute
    se_reduction_ratio: int = 16
    
    # Flags for ablation study
    use_se_attention: bool = True
    use_spatial_attention: bool = True
    
    # =========================================================================
    # Text Encoder (Transformer) Configuration
    # =========================================================================
    # Vocabulary size for questions (will be set after building vocab)
    question_vocab_size: int = 10000  # Placeholder, updated during preprocessing
    
    # Maximum question length (VQA questions are typically short)
    max_question_length: int = 20
    
    # Token embedding dimension
    embed_dim: int = 256
    
    # Transformer configuration
    num_transformer_layers: int = 4
    num_attention_heads: int = 8
    
    # Feed-forward network hidden dimension (typically 4x embed_dim)
    ffn_hidden_dim: int = 1024
    
    # Dropout for regularization
    dropout: float = 0.1
    
    # =========================================================================
    # Cross-Attention Fusion Configuration
    # =========================================================================
    # Cross-attention uses same dimensions as text encoder for compatibility
    cross_attention_heads: int = 8
    
    # Whether to use gating mechanism in fusion
    use_gating: bool = True
    
    # =========================================================================
    # Answer Head Configuration
    # =========================================================================
    # Number of answer classes (top-K most frequent answers)
    num_answers: int = 1000
    
    # Hidden dimension in answer classification MLP
    answer_hidden_dim: int = 512


@dataclass
class TrainingConfig:
    """
    Training loop configuration.
    
    Design Rationale:
    - AdamW optimizer: Better weight decay handling than vanilla Adam
    - CrossEntropyLoss: Standard for multi-class classification
    - Cosine annealing: Smooth LR decay, often outperforms step decay
    - Gradient clipping: Prevents exploding gradients in deep networks
    """
    
    # =========================================================================
    # Dataset Configuration
    # =========================================================================
    # Total Q&A pairs to use (subset for computational constraints)
    total_samples: int = 25000
    
    # Train/validation split ratio
    train_split: float = 0.8  # 80% train, 20% val
    
    # Batch sizes
    train_batch_size: int = 32
    val_batch_size: int = 64  # Larger for faster validation
    
    # Number of data loading workers (0 for Windows compatibility)
    num_workers: int = 0
    
    # Random seed for reproducibility
    seed: int = 42
    
    # =========================================================================
    # Optimizer Configuration
    # =========================================================================
    # Learning rate (carefully tuned for training from scratch)
    learning_rate: float = 1e-4
    
    # AdamW weight decay (regularization)
    weight_decay: float = 0.01
    
    # Adam betas (momentum parameters)
    adam_betas: tuple = (0.9, 0.999)
    
    # =========================================================================
    # LR Scheduler Configuration
    # =========================================================================
    # Use cosine annealing scheduler
    use_scheduler: bool = True
    
    # Minimum learning rate at end of cosine schedule
    min_lr: float = 1e-6
    
    # Warmup epochs before starting cosine decay
    warmup_epochs: int = 2
    
    # =========================================================================
    # Training Loop Configuration
    # =========================================================================
    # Total training epochs
    num_epochs: int = 30
    
    # Gradient clipping max norm
    max_grad_norm: float = 1.0
    
    # Log training metrics every N batches
    log_interval: int = 50
    
    # Validate every N epochs
    val_interval: int = 1
    
    # Save checkpoint every N epochs
    save_interval: int = 5
    
    # Early stopping patience (epochs without improvement)
    patience: int = 10
    
    # =========================================================================
    # Mixed Precision Training
    # =========================================================================
    # Use automatic mixed precision for faster training on GPU
    use_amp: bool = True


@dataclass
class InferenceConfig:
    """Configuration for inference/API serving."""
    
    # Number of top answers to return
    top_k: int = 5
    
    # Confidence threshold for answer
    confidence_threshold: float = 0.1
    
    # Device for inference ('cuda' or 'cpu')
    device: str = "cuda"
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000


# =============================================================================
# Global Configuration Instance
# =============================================================================
# Create default configurations that can be imported throughout the project

PATHS = PathConfig()
MODEL = ModelConfig()
TRAINING = TrainingConfig()
INFERENCE = InferenceConfig()


def get_device() -> str:
    """
    Determine the best available device for training/inference.
    
    Returns:
        str: 'cuda' if GPU available, else 'cpu'
    """
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def print_config():
    """Print all configurations for logging/debugging."""
    print("=" * 60)
    print("VQA System Configuration")
    print("=" * 60)
    print(f"\n[Paths]")
    print(f"  Project Root: {PATHS.project_root}")
    print(f"  COCO Images: {PATHS.coco_images_dir}")
    print(f"  VQA Annotations: {PATHS.vqa_annotations_dir}")
    print(f"\n[Model]")
    print(f"  Image Size: {MODEL.image_size}")
    print(f"  CNN Channels: {MODEL.cnn_channels}")
    print(f"  Embed Dim: {MODEL.embed_dim}")
    print(f"  Transformer Layers: {MODEL.num_transformer_layers}")
    print(f"  Attention Heads: {MODEL.num_attention_heads}")
    print(f"  Num Answers: {MODEL.num_answers}")
    print(f"  SE Attention: {MODEL.use_se_attention}")
    print(f"  Spatial Attention: {MODEL.use_spatial_attention}")
    print(f"\n[Training]")
    print(f"  Total Samples: {TRAINING.total_samples}")
    print(f"  Train Split: {TRAINING.train_split}")
    print(f"  Batch Size: {TRAINING.train_batch_size}")
    print(f"  Learning Rate: {TRAINING.learning_rate}")
    print(f"  Epochs: {TRAINING.num_epochs}")
    print(f"  Device: {get_device()}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
