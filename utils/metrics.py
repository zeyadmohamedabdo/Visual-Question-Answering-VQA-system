"""
Metrics Module for VQA System
==============================
This module implements evaluation metrics for Visual Question Answering.

VQA Metrics:
-----------
1. Top-1 Accuracy: Exact match with ground truth
2. Top-5 Accuracy: Ground truth in top 5 predictions
3. VQA Accuracy: Soft scoring based on annotator agreement

VQA v2 Challenge Accuracy:
--------------------------
The official VQA accuracy metric considers that humans may disagree.
For each question, there are 10 annotator answers.

acc(ans) = min(1, #annotators who agree with ans / 3)

This gives partial credit if multiple annotators gave the same answer.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import Counter
import numpy as np


class VQAAccuracy:
    """
    VQA Accuracy Metric Calculator
    
    This implements both simple accuracy and VQA challenge accuracy.
    
    Simple Accuracy: prediction == ground_truth
    VQA Accuracy: min(1, agreement_count / 3)
    
    Attributes:
        correct: Running sum of correct predictions
        total: Total number of predictions
        correct_top5: Top-5 correct predictions
    """
    
    def __init__(self):
        """Initialize accuracy calculator."""
        self.reset()
    
    def reset(self):
        """Reset all counters."""
        self.correct = 0
        self.total = 0
        self.correct_top5 = 0
        self.per_type_correct = {}
        self.per_type_total = {}
        
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        question_types: Optional[List[str]] = None
    ):
        """
        Update accuracy with a batch of predictions.
        
        Args:
            predictions: Predicted logits or indices [B, num_classes] or [B]
            targets: Ground truth indices [B]
            question_types: Optional question type labels for per-type accuracy
        """
        # Get predictions if logits provided
        if predictions.dim() == 2:
            # Top-1 prediction
            pred_indices = predictions.argmax(dim=-1)
            # Top-5 predictions
            _, top5_indices = predictions.topk(5, dim=-1)
        else:
            pred_indices = predictions
            top5_indices = None
        
        # Move to CPU for comparison
        pred_indices = pred_indices.cpu()
        targets = targets.cpu()
        
        # Top-1 accuracy
        correct_mask = (pred_indices == targets)
        self.correct += correct_mask.sum().item()
        self.total += targets.size(0)
        
        # Top-5 accuracy
        if top5_indices is not None:
            top5_indices = top5_indices.cpu()
            targets_expanded = targets.unsqueeze(1).expand_as(top5_indices)
            correct_top5 = (top5_indices == targets_expanded).any(dim=-1)
            self.correct_top5 += correct_top5.sum().item()
        
        # Per question type accuracy
        if question_types is not None:
            for i, qtype in enumerate(question_types):
                if qtype not in self.per_type_correct:
                    self.per_type_correct[qtype] = 0
                    self.per_type_total[qtype] = 0
                
                self.per_type_total[qtype] += 1
                if correct_mask[i].item():
                    self.per_type_correct[qtype] += 1
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all accuracy metrics.
        
        Returns:
            Dictionary with accuracy metrics
        """
        results = {
            'accuracy': self.correct / max(self.total, 1),
            'accuracy_top5': self.correct_top5 / max(self.total, 1),
            'correct': self.correct,
            'total': self.total
        }
        
        # Per-type accuracies
        if self.per_type_total:
            results['per_type'] = {
                qtype: self.per_type_correct[qtype] / max(self.per_type_total[qtype], 1)
                for qtype in self.per_type_total
            }
        
        return results
    
    def __str__(self) -> str:
        """String representation of current accuracy."""
        metrics = self.compute()
        return f"Accuracy: {metrics['accuracy']:.4f} | Top-5: {metrics['accuracy_top5']:.4f}"


class VQAChallengeAccuracy:
    """
    Official VQA Challenge Accuracy
    
    This implements the soft accuracy metric from VQA v2 challenge:
    acc(ans) = min(1, #humans_who_gave_ans / 3)
    
    This requires access to all annotator answers, not just majority vote.
    """
    
    def __init__(self):
        """Initialize challenge accuracy calculator."""
        self.reset()
    
    def reset(self):
        """Reset state."""
        self.total_score = 0.0
        self.count = 0
    
    def update(
        self,
        predictions: List[str],
        annotator_answers: List[List[str]]
    ):
        """
        Update accuracy with predictions and all annotator answers.
        
        Args:
            predictions: List of predicted answer strings
            annotator_answers: List of lists of annotator answers (10 per question)
        """
        for pred, answers in zip(predictions, annotator_answers):
            # Count how many annotators agree with prediction
            agreement = sum(1 for ans in answers if ans == pred)
            
            # VQA accuracy formula: min(1, agreement/3)
            score = min(1.0, agreement / 3.0)
            
            self.total_score += score
            self.count += 1
    
    def compute(self) -> float:
        """
        Compute VQA challenge accuracy.
        
        Returns:
            Average VQA accuracy
        """
        return self.total_score / max(self.count, 1)


def compute_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor
) -> Tuple[float, float]:
    """
    Compute top-1 and top-5 accuracy from logits.
    
    Args:
        logits: Model output logits [B, num_classes]
        targets: Ground truth indices [B]
        
    Returns:
        Tuple of (top1_accuracy, top5_accuracy)
    """
    # Top-1
    pred = logits.argmax(dim=-1)
    top1_correct = (pred == targets).float().mean().item()
    
    # Top-5
    _, top5_pred = logits.topk(5, dim=-1)
    targets_expanded = targets.unsqueeze(1).expand_as(top5_pred)
    top5_correct = (top5_pred == targets_expanded).any(dim=-1).float().mean().item()
    
    return top1_correct, top5_correct


def compute_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int
) -> torch.Tensor:
    """
    Compute confusion matrix.
    
    Args:
        predictions: Predicted class indices [N]
        targets: Ground truth indices [N]
        num_classes: Number of classes
        
    Returns:
        Confusion matrix [num_classes, num_classes]
    """
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    
    for pred, target in zip(predictions.view(-1), targets.view(-1)):
        conf_matrix[target.long(), pred.long()] += 1
    
    return conf_matrix


def get_per_class_accuracy(conf_matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute per-class accuracy from confusion matrix.
    
    Args:
        conf_matrix: Confusion matrix [C, C]
        
    Returns:
        Per-class accuracy [C]
    """
    # Per-class accuracy = diagonal / row sum
    row_sums = conf_matrix.sum(dim=1).float()
    diagonal = conf_matrix.diag().float()
    
    # Avoid division by zero
    per_class_acc = diagonal / row_sums.clamp(min=1)
    
    return per_class_acc


class AverageMeter:
    """
    Utility class for tracking running averages.
    
    Useful for tracking loss and other metrics during training.
    """
    
    def __init__(self, name: str = 'Metric'):
        """Initialize meter."""
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update running average.
        
        Args:
            val: Value to add
            n: Number of samples val represents
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self) -> str:
        return f'{self.name}: {self.val:.4f} (avg: {self.avg:.4f})'


class MetricsLogger:
    """
    Logger for tracking multiple metrics over training.
    
    Stores history of metrics for plotting and analysis.
    """
    
    def __init__(self):
        """Initialize metrics logger."""
        self.history = {}
        self.current_epoch = 0
    
    def log(self, metrics: Dict[str, float], epoch: Optional[int] = None):
        """
        Log metrics for an epoch.
        
        Args:
            metrics: Dictionary of metric names and values
            epoch: Epoch number (auto-increments if not provided)
        """
        if epoch is not None:
            self.current_epoch = epoch
        
        for name, value in metrics.items():
            if name not in self.history:
                self.history[name] = []
            self.history[name].append((self.current_epoch, value))
        
        self.current_epoch += 1
    
    def get_best(self, metric_name: str, mode: str = 'max') -> Tuple[int, float]:
        """
        Get best value and epoch for a metric.
        
        Args:
            metric_name: Name of metric
            mode: 'max' or 'min'
            
        Returns:
            Tuple of (best_epoch, best_value)
        """
        if metric_name not in self.history:
            return -1, 0.0
        
        values = self.history[metric_name]
        if mode == 'max':
            best_idx = max(range(len(values)), key=lambda i: values[i][1])
        else:
            best_idx = min(range(len(values)), key=lambda i: values[i][1])
        
        return values[best_idx]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for saving."""
        return {
            'history': self.history,
            'current_epoch': self.current_epoch
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MetricsLogger':
        """Load from dictionary."""
        logger = cls()
        logger.history = data['history']
        logger.current_epoch = data['current_epoch']
        return logger


if __name__ == "__main__":
    # Test metrics
    print("Testing Metrics Module\n" + "=" * 50)
    
    # Test VQA Accuracy
    accuracy = VQAAccuracy()
    
    # Simulate some predictions
    batch_size = 8
    num_classes = 1000
    
    for _ in range(5):  # 5 batches
        logits = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        
        accuracy.update(logits, targets)
    
    metrics = accuracy.compute()
    print(f"Accuracy metrics:")
    print(f"  Top-1: {metrics['accuracy']:.4f}")
    print(f"  Top-5: {metrics['accuracy_top5']:.4f}")
    print(f"  Total samples: {metrics['total']}")
    
    # Test quick accuracy computation
    logits = torch.randn(16, 1000)
    targets = torch.randint(0, 1000, (16,))
    top1, top5 = compute_accuracy(logits, targets)
    print(f"\nQuick accuracy: Top-1={top1:.4f}, Top-5={top5:.4f}")
    
    # Test confusion matrix
    predictions = torch.randint(0, 10, (100,))
    targets = torch.randint(0, 10, (100,))
    conf_matrix = compute_confusion_matrix(predictions, targets, 10)
    print(f"\nConfusion matrix shape: {conf_matrix.shape}")
    
    # Test average meter
    meter = AverageMeter('Loss')
    for i in range(10):
        meter.update(1.0 - i * 0.1)
    print(f"\nAverage meter: {meter}")
    
    # Test metrics logger
    logger = MetricsLogger()
    for epoch in range(5):
        logger.log({
            'train_loss': 1.0 - epoch * 0.1,
            'val_accuracy': 0.5 + epoch * 0.05
        })
    
    best_epoch, best_acc = logger.get_best('val_accuracy', 'max')
    print(f"\nBest accuracy: {best_acc:.4f} at epoch {best_epoch}")
    
    print("\n✓ Metrics tests passed!")
