"""
Inference Module for VQA System
================================
This module provides inference utilities for the VQA model.

Features:
- Load trained model from checkpoint
- Preprocess single images
- Tokenize questions
- Run inference and decode predictions

This is used by the API for serving predictions.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from io import BytesIO

import torch
import torch.nn.functional as F
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.vqa_model import VQAModel, load_vqa_model
from data.preprocess import get_inference_transforms, preprocess_image_from_bytes
from data.build_vocab import AnswerVocabulary
from utils.tokenizer import Tokenizer
from utils.config import PATHS, MODEL, INFERENCE, get_device


class VQAInference:
    """
    Inference engine for VQA model.
    
    This class handles:
    - Model loading and management
    - Image preprocessing
    - Question tokenization
    - Prediction and answer decoding
    
    Designed for production use with the FastAPI backend.
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        question_vocab_path: Optional[str] = None,
        answer_vocab_path: Optional[str] = None
    ):
        """
        Initialize inference engine.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device for inference ('cuda' or 'cpu')
            question_vocab_path: Path to question vocabulary
            answer_vocab_path: Path to answer vocabulary
        """
        # Set device
        self.device = device or get_device()
        
        # Set paths
        self.checkpoint_path = checkpoint_path or PATHS.best_model_path
        self.question_vocab_path = question_vocab_path or PATHS.question_vocab_file
        self.answer_vocab_path = answer_vocab_path or PATHS.vocab_file
        
        # Initialize components as None (lazy loading)
        self.model = None
        self.tokenizer = None
        self.answer_vocab = None
        self.transform = None
        
        # Track if loaded
        self._is_loaded = False
        
    def load(self):
        """
        Load model and vocabularies.
        
        Call this before running inference.
        """
        if self._is_loaded:
            return
        
        print(f"[Inference] Loading model from {self.checkpoint_path}...")
        
        # Load model
        if os.path.exists(self.checkpoint_path):
            self.model = load_vqa_model(self.checkpoint_path, self.device)
        else:
            # Create default model if no checkpoint
            print("[Inference] No checkpoint found, creating default model...")
            self.model = VQAModel(
                vocab_size=MODEL.question_vocab_size,
                num_answers=MODEL.num_answers
            ).to(self.device)
        
        self.model.eval()
        
        # Load tokenizer
        if os.path.exists(self.question_vocab_path):
            self.tokenizer = Tokenizer()
            self.tokenizer.load(self.question_vocab_path)
            print(f"[Inference] Loaded tokenizer with {self.tokenizer.vocab_size} tokens")
        else:
            # Create simple tokenizer
            print("[Inference] Creating default tokenizer...")
            self.tokenizer = Tokenizer(max_length=MODEL.max_question_length)
            # Build minimal vocab
            self.tokenizer.build_vocab([
                "what is this", "what color", "how many",
                "is there", "where is", "what type"
            ], min_freq=1)
        
        # Load answer vocabulary
        if os.path.exists(self.answer_vocab_path):
            self.answer_vocab = AnswerVocabulary()
            self.answer_vocab.load(self.answer_vocab_path)
            print(f"[Inference] Loaded {self.answer_vocab.num_answers} answers")
        else:
            # Create simple answer vocab
            print("[Inference] Creating default answer vocabulary...")
            self.answer_vocab = AnswerVocabulary(num_answers=MODEL.num_answers)
            # Map indices to generic answers
            for i in range(MODEL.num_answers):
                self.answer_vocab.idx2answer[i] = f"answer_{i}"
        
        # Create image transform
        self.transform = get_inference_transforms(MODEL.image_size)
        
        self._is_loaded = True
        print("[Inference] Ready for predictions!")
    
    def preprocess_image(
        self,
        image: Union[str, bytes, Image.Image]
    ) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: Image path, bytes, or PIL Image
            
        Returns:
            Preprocessed tensor [1, 3, 224, 224]
        """
        if isinstance(image, str):
            # Load from path
            pil_image = Image.open(image)
        elif isinstance(image, bytes):
            # Load from bytes
            pil_image = Image.open(BytesIO(image))
        else:
            pil_image = image
        
        # Convert to RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Apply transforms
        tensor = self.transform(pil_image)
        
        # Add batch dimension
        return tensor.unsqueeze(0)
    
    def preprocess_question(self, question: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize and encode question.
        
        Args:
            question: Question string
            
        Returns:
            Tuple of (token_ids, attention_mask) with batch dimension
        """
        token_ids, attention_mask = self.tokenizer.encode(
            question,
            add_special_tokens=True,
            padding=True,
            truncation=True
        )
        
        # Convert to tensors with batch dimension
        token_ids = torch.tensor([token_ids], dtype=torch.long)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long)
        
        return token_ids, attention_mask
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[str, bytes, Image.Image],
        question: str,
        top_k: int = 5
    ) -> Dict:
        """
        Run inference for a single image-question pair.
        
        Args:
            image: Image (path, bytes, or PIL Image)
            question: Question string
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions:
            - question: Original question
            - answers: List of top-k answers with probabilities
            - top_answer: Best answer
            - confidence: Confidence of top answer
        """
        # Ensure loaded
        if not self._is_loaded:
            self.load()
        
        # Preprocess inputs
        image_tensor = self.preprocess_image(image).to(self.device)
        token_ids, attention_mask = self.preprocess_question(question)
        token_ids = token_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Forward pass
        logits, _ = self.model(image_tensor, token_ids, attention_mask)
        
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Get top-k predictions
        top_probs, top_indices = probs.topk(top_k, dim=-1)
        
        # Decode answers
        answers = []
        for i in range(top_k):
            idx = top_indices[0, i].item()
            prob = top_probs[0, i].item()
            answer_str = self.answer_vocab.decode(idx)
            answers.append({
                'answer': answer_str,
                'probability': prob,
                'index': idx
            })
        
        return {
            'question': question,
            'answers': answers,
            'top_answer': answers[0]['answer'],
            'confidence': answers[0]['probability']
        }
    
    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[str, bytes, Image.Image]],
        questions: List[str],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Run inference for a batch of image-question pairs.
        
        Args:
            images: List of images
            questions: List of questions
            top_k: Top-k predictions per sample
            
        Returns:
            List of prediction dictionaries
        """
        if len(images) != len(questions):
            raise ValueError("Number of images must match number of questions")
        
        # Ensure loaded
        if not self._is_loaded:
            self.load()
        
        # Preprocess all inputs
        image_tensors = torch.cat([
            self.preprocess_image(img) for img in images
        ], dim=0).to(self.device)
        
        all_token_ids = []
        all_attention_masks = []
        for question in questions:
            token_ids, attention_mask = self.preprocess_question(question)
            all_token_ids.append(token_ids)
            all_attention_masks.append(attention_mask)
        
        token_ids = torch.cat(all_token_ids, dim=0).to(self.device)
        attention_masks = torch.cat(all_attention_masks, dim=0).to(self.device)
        
        # Forward pass
        logits, _ = self.model(image_tensors, token_ids, attention_masks)
        
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = probs.topk(top_k, dim=-1)
        
        # Decode all answers
        results = []
        for batch_idx in range(len(images)):
            answers = []
            for i in range(top_k):
                idx = top_indices[batch_idx, i].item()
                prob = top_probs[batch_idx, i].item()
                answer_str = self.answer_vocab.decode(idx)
                answers.append({
                    'answer': answer_str,
                    'probability': prob,
                    'index': idx
                })
            
            results.append({
                'question': questions[batch_idx],
                'answers': answers,
                'top_answer': answers[0]['answer'],
                'confidence': answers[0]['probability']
            })
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        if not self._is_loaded:
            self.load()
        
        param_counts = self.model.get_num_parameters()
        
        return {
            'device': str(self.device),
            'vocab_size': self.tokenizer.vocab_size,
            'num_answers': self.answer_vocab.num_answers,
            'parameters': param_counts,
            'config': self.model.config
        }


# Global inference instance (for API use)
_inference_instance: Optional[VQAInference] = None


def get_inference_engine() -> VQAInference:
    """
    Get or create global inference engine.
    
    Returns:
        VQAInference instance
    """
    global _inference_instance
    
    if _inference_instance is None:
        _inference_instance = VQAInference()
        _inference_instance.load()
    
    return _inference_instance


if __name__ == "__main__":
    # Test inference
    print("Testing VQA Inference\n" + "=" * 50)
    
    # Create inference engine
    engine = VQAInference()
    engine.load()
    
    # Get model info
    info = engine.get_model_info()
    print(f"\nModel info:")
    print(f"  Device: {info['device']}")
    print(f"  Vocab size: {info['vocab_size']}")
    print(f"  Num answers: {info['num_answers']}")
    print(f"  Parameters: {info['parameters']['total']:,}")
    
    # Test with a dummy image
    print("\nTesting with dummy image...")
    dummy_image = Image.new('RGB', (640, 480), color=(100, 150, 200))
    question = "What color is this?"
    
    result = engine.predict(dummy_image, question)
    
    print(f"\nQuestion: {result['question']}")
    print(f"Top answer: {result['top_answer']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"\nTop 5 answers:")
    for i, ans in enumerate(result['answers']):
        print(f"  {i+1}. {ans['answer']}: {ans['probability']:.4f}")
    
    print("\n✓ Inference tests passed!")
