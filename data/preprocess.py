"""
Preprocessing Module for VQA System
====================================
This module handles image and text preprocessing for training and inference.

Image Preprocessing:
-------------------
- Resize to 224x224 (standard for ImageNet-style CNNs)
- Normalize with ImageNet statistics (even though we don't use pretrained weights,
  these are well-studied values that help with training stability)
- Data augmentation for training (horizontal flip, color jitter)

Text Preprocessing:
------------------
- Handled primarily by tokenizer.py
- This module provides additional utilities for question normalization
"""

import torch
from torchvision import transforms
from PIL import Image
from typing import Tuple, Callable, Optional
import numpy as np


# =============================================================================
# Image Normalization Constants
# =============================================================================
# Using ImageNet statistics because:
# 1. They are well-studied and provide good training dynamics
# 2. COCO images have similar statistics
# 3. Helps with numerical stability during training

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(
    image_size: int = 224,
    use_augmentation: bool = True
) -> transforms.Compose:
    """
    Get image transforms for training.
    
    Training transforms include data augmentation to:
    1. Increase effective training data size
    2. Improve generalization
    3. Handle various image orientations and lighting
    
    Args:
        image_size: Target image size (square)
        use_augmentation: Whether to apply augmentation
        
    Returns:
        Composed transform pipeline
    
    Transform Pipeline:
        Input: PIL Image (any size, RGB)
        1. Resize to (image_size, image_size)
        2. RandomHorizontalFlip (50% chance)
        3. ColorJitter (brightness, contrast, saturation)
        4. ToTensor [H, W, C] -> [C, H, W], scale to [0, 1]
        5. Normalize with ImageNet mean/std
        Output: Tensor [3, image_size, image_size]
    """
    if use_augmentation:
        transform = transforms.Compose([
            # Resize with slight scale variation for robustness
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop((image_size, image_size)),
            
            # Horizontal flip - doesn't change answers for most questions
            transforms.RandomHorizontalFlip(p=0.5),
            
            # Color augmentation - helps with lighting variations
            # Kept mild to avoid creating unrealistic images
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            
            # Convert to tensor and normalize
            transforms.ToTensor(),  # [C, H, W], values in [0, 1]
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    
    return transform


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get image transforms for validation/inference.
    
    No data augmentation - we want deterministic evaluation.
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed transform pipeline
        
    Transform Pipeline:
        Input: PIL Image (any size, RGB)
        1. Resize to (image_size, image_size)
        2. ToTensor
        3. Normalize
        Output: Tensor [3, image_size, image_size]
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_inference_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get image transforms for single-image inference.
    
    Same as validation transforms but can be easily customized
    for production use (e.g., different error handling).
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed transform pipeline
    """
    return get_val_transforms(image_size)


def load_and_preprocess_image(
    image_path: str,
    transform: transforms.Compose,
    return_original: bool = False
) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
    """
    Load image from path and apply transforms.
    
    Args:
        image_path: Path to image file
        transform: Transform pipeline to apply
        return_original: If True, also return original numpy array
        
    Returns:
        Tuple of (transformed_tensor, original_array or None)
        
    Shape:
        transformed_tensor: [3, H, W] where H=W=image_size
        original_array: [H, W, 3] if return_original else None
    """
    # Load image
    image = Image.open(image_path)
    
    # Convert to RGB if needed (handle grayscale or RGBA)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    original = None
    if return_original:
        original = np.array(image)
    
    # Apply transforms
    transformed = transform(image)
    
    return transformed, original


def preprocess_image_from_bytes(
    image_bytes: bytes,
    transform: transforms.Compose
) -> torch.Tensor:
    """
    Load image from bytes (for API use) and apply transforms.
    
    Args:
        image_bytes: Raw image bytes (e.g., from file upload)
        transform: Transform pipeline
        
    Returns:
        Transformed tensor [3, H, W]
    """
    from io import BytesIO
    
    image = Image.open(BytesIO(image_bytes))
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return transform(image)


def denormalize_image(
    tensor: torch.Tensor,
    mean: list = IMAGENET_MEAN,
    std: list = IMAGENET_STD
) -> torch.Tensor:
    """
    Denormalize image tensor for visualization.
    
    Reverses the normalization: x = (normalized * std) + mean
    
    Args:
        tensor: Normalized image tensor [C, H, W] or [B, C, H, W]
        mean: Mean values used for normalization
        std: Std values used for normalization
        
    Returns:
        Denormalized tensor with values in [0, 1]
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    denormalized = tensor * std + mean
    return torch.clamp(denormalized, 0, 1)


# =============================================================================
# Text Preprocessing Utilities
# =============================================================================

def normalize_question(question: str) -> str:
    """
    Normalize question text for consistency.
    
    This is a lighter preprocessing than tokenizer.preprocess(),
    used for display and logging purposes.
    
    Args:
        question: Raw question string
        
    Returns:
        Normalized question
    """
    # Ensure proper spacing after punctuation
    question = question.strip()
    
    # Ensure ends with question mark
    if not question.endswith('?'):
        question = question + '?'
    
    # Capitalize first letter
    question = question[0].upper() + question[1:] if question else question
    
    return question


def validate_question(question: str, min_words: int = 2) -> Tuple[bool, str]:
    """
    Validate question text.
    
    Args:
        question: Input question
        min_words: Minimum number of words required
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not question or not question.strip():
        return False, "Question cannot be empty"
    
    words = question.strip().split()
    if len(words) < min_words:
        return False, f"Question must have at least {min_words} words"
    
    return True, ""


# =============================================================================
# Collate Function for DataLoader
# =============================================================================

def vqa_collate_fn(batch):
    """
    Custom collate function for VQA DataLoader.
    
    Handles batching of:
    - Images: Stack into tensor
    - Token IDs: Already padded, stack into tensor
    - Attention masks: Stack into tensor
    - Answer indices: Stack into tensor
    
    Args:
        batch: List of (image, token_ids, attention_mask, answer_idx) tuples
        
    Returns:
        Dict with batched tensors:
        - 'images': [B, 3, H, W]
        - 'token_ids': [B, seq_len]
        - 'attention_mask': [B, seq_len]
        - 'answers': [B]
    """
    images = torch.stack([item[0] for item in batch])
    token_ids = torch.stack([item[1] for item in batch])
    attention_masks = torch.stack([item[2] for item in batch])
    answers = torch.tensor([item[3] for item in batch], dtype=torch.long)
    
    return {
        'images': images,           # [B, 3, 224, 224]
        'token_ids': token_ids,     # [B, max_length]
        'attention_mask': attention_masks,  # [B, max_length]
        'answers': answers          # [B]
    }


if __name__ == "__main__":
    # Demo transforms
    print("Training Transforms:")
    train_tf = get_train_transforms(224, use_augmentation=True)
    print(train_tf)
    
    print("\nValidation Transforms:")
    val_tf = get_val_transforms(224)
    print(val_tf)
    
    # Demo with a dummy image
    dummy_image = Image.new('RGB', (640, 480), color=(128, 128, 128))
    
    transformed = train_tf(dummy_image)
    print(f"\nTransformed image shape: {transformed.shape}")
    print(f"Value range: [{transformed.min():.3f}, {transformed.max():.3f}]")
    
    # Demo denormalization
    denorm = denormalize_image(transformed)
    print(f"Denormalized range: [{denorm.min():.3f}, {denorm.max():.3f}]")
