"""
Evaluation Script for VQA System
=================================
This module implements comprehensive evaluation of trained VQA models.

Evaluation includes:
1. Accuracy metrics (Top-1, Top-5)
2. Per-question-type analysis
3. Confusion matrix for common answers
4. Attention visualization examples
5. Error analysis

Usage:
------
python training/evaluate.py --checkpoint checkpoints/best_model.pth
python training/evaluate.py --checkpoint checkpoints/best_model.pth --visualize
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.vqa_model import VQAModel, load_vqa_model
from data.dataset import VQADataset, create_demo_loaders
from data.build_vocab import AnswerVocabulary
from utils.tokenizer import Tokenizer
from utils.config import PATHS, MODEL, TRAINING, get_device
from utils.metrics import VQAAccuracy, compute_confusion_matrix, get_per_class_accuracy


class Evaluator:
    """
    Evaluation manager for VQA model.
    
    This class handles:
    - Loading trained model
    - Running evaluation on validation set
    - Computing various metrics
    - Generating analysis reports
    """
    
    def __init__(
        self,
        model: VQAModel,
        val_loader: DataLoader,
        answer_vocab: Optional[AnswerVocabulary] = None,
        device: str = 'cuda'
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained VQA model
            val_loader: Validation data loader
            answer_vocab: Answer vocabulary for decoding
            device: Evaluation device
        """
        self.model = model.to(device)
        self.model.eval()
        self.val_loader = val_loader
        self.answer_vocab = answer_vocab
        self.device = device
        
    @torch.no_grad()
    def evaluate(self) -> Dict:
        """
        Run full evaluation.
        
        Returns:
            Dictionary with all evaluation metrics
        """
        accuracy = VQAAccuracy()
        all_predictions = []
        all_targets = []
        all_logits = []
        
        print("\n[Evaluator] Running evaluation...")
        
        for batch in tqdm(self.val_loader, desc='Evaluating'):
            images = batch['images'].to(self.device)
            token_ids = batch['token_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            targets = batch['answers'].to(self.device)
            
            # Forward pass
            logits, _ = self.model(images, token_ids, attention_mask)
            
            # Update accuracy
            accuracy.update(logits, targets)
            
            # Store predictions and targets
            predictions = logits.argmax(dim=-1)
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            all_logits.append(logits.cpu())
        
        # Concatenate all results
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        all_logits = torch.cat(all_logits)
        
        # Compute metrics
        metrics = accuracy.compute()
        
        # Confusion matrix for top answers
        num_top_answers = min(100, MODEL.num_answers)
        conf_matrix = compute_confusion_matrix(
            all_predictions, all_targets, MODEL.num_answers
        )
        
        # Per-class accuracy for top answers
        per_class_acc = get_per_class_accuracy(conf_matrix)
        
        # Find most common errors
        errors = self._analyze_errors(all_predictions, all_targets, all_logits)
        
        results = {
            'accuracy': metrics['accuracy'],
            'accuracy_top5': metrics['accuracy_top5'],
            'total_samples': metrics['total'],
            'correct': metrics['correct'],
            'per_class_accuracy': per_class_acc[:num_top_answers].tolist(),
            'common_errors': errors
        }
        
        return results
    
    def _analyze_errors(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        logits: torch.Tensor,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Analyze most common prediction errors.
        
        Args:
            predictions: Model predictions [N]
            targets: Ground truth [N]
            logits: Model logits [N, C]
            top_k: Number of top errors to return
            
        Returns:
            List of error analysis dicts
        """
        # Find incorrect predictions
        incorrect_mask = predictions != targets
        incorrect_indices = torch.where(incorrect_mask)[0]
        
        if len(incorrect_indices) == 0:
            return []
        
        # Count (predicted, actual) pairs
        error_pairs = {}
        for idx in incorrect_indices:
            pred = predictions[idx].item()
            target = targets[idx].item()
            pair = (pred, target)
            error_pairs[pair] = error_pairs.get(pair, 0) + 1
        
        # Sort by frequency
        sorted_errors = sorted(error_pairs.items(), key=lambda x: x[1], reverse=True)
        
        # Decode if vocabulary available
        errors = []
        for (pred, target), count in sorted_errors[:top_k]:
            error_info = {
                'predicted_idx': pred,
                'target_idx': target,
                'count': count
            }
            
            if self.answer_vocab:
                error_info['predicted'] = self.answer_vocab.decode(pred)
                error_info['target'] = self.answer_vocab.decode(target)
            
            errors.append(error_info)
        
        return errors
    
    @torch.no_grad()
    def get_sample_predictions(
        self,
        num_samples: int = 10
    ) -> List[Dict]:
        """
        Get predictions for sample images.
        
        Args:
            num_samples: Number of samples to return
            
        Returns:
            List of sample predictions with metadata
        """
        samples = []
        
        for batch in self.val_loader:
            images = batch['images'].to(self.device)
            token_ids = batch['token_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            targets = batch['answers']
            
            # Get predictions
            top_indices, top_probs = self.model.predict(
                images, token_ids, attention_mask, top_k=5
            )
            
            for i in range(len(images)):
                if len(samples) >= num_samples:
                    break
                
                sample = {
                    'target_idx': targets[i].item(),
                    'predictions': top_indices[i].tolist(),
                    'probabilities': top_probs[i].tolist(),
                    'correct': targets[i].item() == top_indices[i, 0].item()
                }
                
                if self.answer_vocab:
                    sample['target'] = self.answer_vocab.decode(targets[i].item())
                    sample['predicted_answers'] = [
                        self.answer_vocab.decode(idx) for idx in top_indices[i].tolist()
                    ]
                
                samples.append(sample)
            
            if len(samples) >= num_samples:
                break
        
        return samples
    
    def generate_report(self, results: Dict, output_path: str):
        """
        Generate evaluation report.
        
        Args:
            results: Evaluation results dictionary
            output_path: Path to save report
        """
        report_lines = [
            "=" * 60,
            "VQA Model Evaluation Report",
            "=" * 60,
            "",
            "OVERALL METRICS",
            "-" * 40,
            f"Top-1 Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)",
            f"Top-5 Accuracy: {results['accuracy_top5']:.4f} ({results['accuracy_top5']*100:.2f}%)",
            f"Total Samples: {results['total_samples']}",
            f"Correct Predictions: {results['correct']}",
            "",
            "COMMON ERRORS",
            "-" * 40,
        ]
        
        for i, error in enumerate(results.get('common_errors', [])[:10]):
            if 'predicted' in error:
                report_lines.append(
                    f"{i+1}. Predicted '{error['predicted']}' instead of '{error['target']}': "
                    f"{error['count']} times"
                )
            else:
                report_lines.append(
                    f"{i+1}. Predicted {error['predicted_idx']} instead of {error['target_idx']}: "
                    f"{error['count']} times"
                )
        
        report_lines.extend([
            "",
            "=" * 60
        ])
        
        report_text = "\n".join(report_lines)
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\nReport saved to {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate VQA model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--images-dir', type=str, default=PATHS.coco_images_dir,
                        help='Path to COCO images')
    parser.add_argument('--questions-file', type=str,
                        default=os.path.join(PATHS.vqa_annotations_dir, 'v2_OpenEnded_mscoco_val2014_questions.json'),
                        help='Path to VQA questions')
    parser.add_argument('--annotations-file', type=str,
                        default=os.path.join(PATHS.vqa_annotations_dir, 'v2_mscoco_val2014_annotations.json'),
                        help='Path to VQA annotations')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Evaluation batch size')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--demo', action='store_true',
                        help='Use demo data')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate attention visualizations')
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    device = get_device()
    print(f"\n[Evaluator] Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\n[Evaluator] Loading model from {args.checkpoint}...")
    model = load_vqa_model(args.checkpoint, device)
    
    # Load answer vocabulary if available
    answer_vocab = None
    if Path(PATHS.vocab_file).exists():
        answer_vocab = AnswerVocabulary()
        answer_vocab.load(PATHS.vocab_file)
    
    # Create validation data loader
    if args.demo:
        print("[Evaluator] Using demo data...")
        _, val_loader = create_demo_loaders(
            num_train=100,
            num_val=500,
            batch_size=args.batch_size
        )
    else:
        print("[Evaluator] Loading validation data...")
        try:
            # Load tokenizer
            tokenizer = Tokenizer()
            tokenizer.load(PATHS.question_vocab_file)
            
            val_dataset = VQADataset(
                images_dir=args.images_dir,
                questions_file=args.questions_file,
                annotations_file=args.annotations_file,
                tokenizer=tokenizer,
                answer_vocab=answer_vocab,
                is_training=False
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0
            )
        except FileNotFoundError:
            print("[Evaluator] Data files not found, using demo mode...")
            _, val_loader = create_demo_loaders(
                num_train=100,
                num_val=500,
                batch_size=args.batch_size
            )
    
    # Create evaluator and run evaluation
    evaluator = Evaluator(model, val_loader, answer_vocab, device)
    results = evaluator.evaluate()
    
    # Get sample predictions
    samples = evaluator.get_sample_predictions(num_samples=20)
    results['sample_predictions'] = samples
    
    # Save results
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Evaluator] Results saved to {results_path}")
    
    # Generate report
    report_path = output_dir / 'evaluation_report.txt'
    evaluator.generate_report(results, str(report_path))


if __name__ == '__main__':
    main()
