"""
VQA Dataset Module
==================
This module implements the PyTorch Dataset for Visual Question Answering.

Dataset Structure:
-----------------
- Images: COCO val2017 (~5000 images)
- Questions & Answers: VQA v2 validation set
- Subset: ~25,000 Q&A pairs (as per project constraints)

Data Loading Strategy:
---------------------
1. Load and parse VQA annotations on initialization
2. Filter to questions with valid answers (in top-1000 vocabulary)
3. Create 80/20 train/validation split
4. Load images on-demand (memory efficient)
5. Apply transforms per-sample
"""

import os
import json
import random
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocess import get_train_transforms, get_val_transforms, vqa_collate_fn
from data.build_vocab import AnswerVocabulary
from utils.tokenizer import Tokenizer, create_tokenizer_from_questions
from utils.config import PATHS, MODEL, TRAINING


class VQADataset(Dataset):
    """
    PyTorch Dataset for VQA task.
    
    This dataset loads COCO images and VQA v2 question-answer pairs,
    applies preprocessing, and returns tensors ready for model training.
    
    Attributes:
        samples (List[Dict]): List of QA samples with image paths
        transform: Image transform pipeline
        tokenizer (Tokenizer): Question tokenizer
        answer_vocab (AnswerVocabulary): Answer vocabulary for classification
        
    Returns per sample:
        image_tensor: [3, 224, 224] normalized image
        token_ids: [max_length] padded token indices
        attention_mask: [max_length] attention mask (1=real, 0=pad)
        answer_idx: int - answer class index
    """
    
    def __init__(
        self,
        images_dir: str,
        questions_file: str,
        annotations_file: str,
        tokenizer: Optional[Tokenizer] = None,
        answer_vocab: Optional[AnswerVocabulary] = None,
        transform=None,
        max_samples: Optional[int] = None,
        is_training: bool = True,
        load_answers: bool = True
    ):
        """
        Initialize VQA Dataset.
        
        Args:
            images_dir: Path to COCO images directory
            questions_file: Path to VQA questions JSON
            annotations_file: Path to VQA annotations JSON
            tokenizer: Pre-built tokenizer (if None, will be built)
            answer_vocab: Pre-built answer vocabulary (if None, will be built)
            transform: Image transforms (if None, uses defaults)
            max_samples: Limit number of samples (for subset training)
            is_training: If True, use training transforms
            load_answers: If True, load and filter by valid answers
        """
        self.images_dir = Path(images_dir)
        self.is_training = is_training
        self.load_answers = load_answers
        
        # Set transforms
        if transform is None:
            if is_training:
                self.transform = get_train_transforms(MODEL.image_size)
            else:
                self.transform = get_val_transforms(MODEL.image_size)
        else:
            self.transform = transform
        
        # Load questions
        print(f"[VQADataset] Loading questions from {questions_file}")
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
        
        # Create question_id to question mapping
        self.questions = {
            q['question_id']: q 
            for q in questions_data['questions']
        }
        print(f"[VQADataset] Loaded {len(self.questions)} questions")
        
        # Load annotations if available
        self.annotations = {}
        if load_answers and annotations_file and os.path.exists(annotations_file):
            print(f"[VQADataset] Loading annotations from {annotations_file}")
            with open(annotations_file, 'r', encoding='utf-8') as f:
                annotations_data = json.load(f)
            self.annotations = {
                ann['question_id']: ann 
                for ann in annotations_data['annotations']
            }
            print(f"[VQADataset] Loaded {len(self.annotations)} annotations")
        
        # Build or assign answer vocabulary
        if answer_vocab is not None:
            self.answer_vocab = answer_vocab
        elif load_answers and self.annotations:
            self.answer_vocab = AnswerVocabulary(num_answers=MODEL.num_answers)
            self.answer_vocab.build_from_qa_pairs(
                [{'answer': ann['multiple_choice_answer']} 
                 for ann in self.annotations.values()]
            )
        else:
            self.answer_vocab = None
        
        # Build samples list
        self.samples = self._build_samples(max_samples)
        print(f"[VQADataset] Created {len(self.samples)} valid samples")
        
        # Build or assign tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            questions_list = [s['question'] for s in self.samples]
            self.tokenizer = create_tokenizer_from_questions(
                questions_list,
                max_length=MODEL.max_question_length,
                vocab_size=MODEL.question_vocab_size
            )
    
    def _build_samples(self, max_samples: Optional[int]) -> List[Dict]:
        """
        Build list of valid samples.
        
        Filters to samples where:
        1. Image file exists
        2. Answer is in vocabulary (if using answers)
        
        Args:
            max_samples: Maximum number of samples to include
            
        Returns:
            List of sample dictionaries
        """
        samples = []
        
        for qid, question_info in self.questions.items():
            # Get image path
            image_id = question_info['image_id']
            # COCO image filename format: 000000XXXXXX.jpg (12 digits)
            image_filename = f"{image_id:012d}.jpg"
            image_path = self.images_dir / image_filename
            
            # Skip if image doesn't exist
            if not image_path.exists():
                continue
            
            # Get answer if loading answers
            answer = None
            answer_idx = -1
            if self.load_answers and qid in self.annotations:
                answer = self.annotations[qid]['multiple_choice_answer']
                if self.answer_vocab is not None:
                    answer_idx = self.answer_vocab.encode(answer)
                    # Skip if answer not in vocabulary
                    if answer_idx == -1:
                        continue
            
            samples.append({
                'question_id': qid,
                'image_id': image_id,
                'image_path': str(image_path),
                'question': question_info['question'],
                'answer': answer,
                'answer_idx': answer_idx
            })
            
            # Check max samples
            if max_samples and len(samples) >= max_samples:
                break
        
        return samples
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of:
            - image_tensor: [3, 224, 224]
            - token_ids: [max_length]
            - attention_mask: [max_length]
            - answer_idx: int
        """
        sample = self.samples[idx]
        
        # Load and transform image
        image = Image.open(sample['image_path'])
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_tensor = self.transform(image)
        
        # Tokenize question
        token_ids, attention_mask = self.tokenizer.encode(
            sample['question'],
            add_special_tokens=True,
            padding=True,
            truncation=True
        )
        
        # Convert to tensors
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        
        return (
            image_tensor,      # [3, 224, 224]
            token_ids,         # [max_length]
            attention_mask,    # [max_length]
            sample['answer_idx']
        )
    
    def get_sample_info(self, idx: int) -> Dict:
        """
        Get raw sample information (for debugging/visualization).
        
        Args:
            idx: Sample index
            
        Returns:
            Sample dictionary with all metadata
        """
        return self.samples[idx]


def create_train_val_loaders(
    images_dir: str,
    questions_file: str,
    annotations_file: str,
    total_samples: int = 25000,
    train_split: float = 0.8,
    train_batch_size: int = 32,
    val_batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, Tokenizer, AnswerVocabulary]:
    """
    Create train and validation data loaders with shared tokenizers.
    
    This function:
    1. Creates a single dataset with all samples
    2. Builds shared tokenizer and answer vocabulary
    3. Splits into train/val with fixed seed for reproducibility
    4. Creates DataLoaders with appropriate transforms
    
    Args:
        images_dir: Path to COCO images
        questions_file: Path to VQA questions
        annotations_file: Path to VQA annotations
        total_samples: Total number of samples to use
        train_split: Fraction for training (rest is validation)
        train_batch_size: Training batch size
        val_batch_size: Validation batch size
        num_workers: DataLoader workers (0 for Windows)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_loader, val_loader, tokenizer, answer_vocab)
    """
    # Set seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Create full dataset to build vocabularies
    print("[DataLoader] Building full dataset for vocabulary extraction...")
    full_dataset = VQADataset(
        images_dir=images_dir,
        questions_file=questions_file,
        annotations_file=annotations_file,
        max_samples=total_samples,
        is_training=True  # Doesn't matter for vocab building
    )
    
    # Get tokenizer and answer vocab from full dataset
    tokenizer = full_dataset.tokenizer
    answer_vocab = full_dataset.answer_vocab
    
    # Split indices
    all_indices = list(range(len(full_dataset)))
    random.shuffle(all_indices)
    
    split_idx = int(len(all_indices) * train_split)
    train_indices = all_indices[:split_idx]
    val_indices = all_indices[split_idx:]
    
    print(f"[DataLoader] Train samples: {len(train_indices)}")
    print(f"[DataLoader] Val samples: {len(val_indices)}")
    
    # Create subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    # Override transforms for validation subset
    # Note: Since Subset wraps the dataset, we need to be careful here
    # For cleaner separation, we create separate datasets
    
    # Actually, let's create separate datasets with shared vocabs
    train_dataset = VQADataset(
        images_dir=images_dir,
        questions_file=questions_file,
        annotations_file=annotations_file,
        tokenizer=tokenizer,
        answer_vocab=answer_vocab,
        max_samples=total_samples,
        is_training=True
    )
    
    # Filter to train indices
    train_dataset.samples = [train_dataset.samples[i] for i in train_indices if i < len(train_dataset.samples)]
    
    val_dataset = VQADataset(
        images_dir=images_dir,
        questions_file=questions_file,
        annotations_file=annotations_file,
        tokenizer=tokenizer,
        answer_vocab=answer_vocab,
        max_samples=total_samples,
        is_training=False  # No augmentation
    )
    val_dataset.samples = [val_dataset.samples[i] for i in val_indices if i < len(val_dataset.samples)]
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=vqa_collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=vqa_collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, tokenizer, answer_vocab


# =============================================================================
# Demo Data Generator (for testing without real data)
# =============================================================================

class DemoVQADataset(Dataset):
    """
    Demo dataset for testing pipeline without real COCO/VQA data.
    
    Generates random images and questions for architecture testing.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        image_size: int = 224,
        max_question_length: int = 20,
        vocab_size: int = 1000,
        num_answers: int = 1000
    ):
        """
        Initialize demo dataset.
        
        Args:
            num_samples: Number of fake samples
            image_size: Image dimension
            max_question_length: Max tokens in question
            vocab_size: Question vocabulary size
            num_answers: Number of answer classes
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.max_question_length = max_question_length
        self.vocab_size = vocab_size
        self.num_answers = num_answers
        
        print(f"[DemoDataset] Created with {num_samples} fake samples")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int):
        """Generate random sample."""
        # Random normalized image
        image = torch.randn(3, self.image_size, self.image_size)
        
        # Random token IDs
        token_ids = torch.randint(0, self.vocab_size, (self.max_question_length,))
        
        # Attention mask (random length)
        seq_len = random.randint(5, self.max_question_length)
        attention_mask = torch.zeros(self.max_question_length, dtype=torch.long)
        attention_mask[:seq_len] = 1
        
        # Random answer
        answer_idx = random.randint(0, self.num_answers - 1)
        
        return image, token_ids, attention_mask, answer_idx


def create_demo_loaders(
    num_train: int = 5000,
    num_val: int = 1000,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """
    Create demo DataLoaders for testing.
    
    Args:
        num_train: Training samples
        num_val: Validation samples
        batch_size: Batch size
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = DemoVQADataset(num_samples=num_train)
    val_dataset = DemoVQADataset(num_samples=num_val)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=vqa_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=vqa_collate_fn
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test with demo data
    print("Testing with demo dataset...")
    train_loader, val_loader = create_demo_loaders(
        num_train=100,
        num_val=20,
        batch_size=8
    )
    
    # Get a batch
    batch = next(iter(train_loader))
    print(f"\nBatch contents:")
    print(f"  Images shape: {batch['images'].shape}")
    print(f"  Token IDs shape: {batch['token_ids'].shape}")
    print(f"  Attention mask shape: {batch['attention_mask'].shape}")
    print(f"  Answers shape: {batch['answers'].shape}")
