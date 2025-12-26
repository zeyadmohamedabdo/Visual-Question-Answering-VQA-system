"""
Vocabulary Builder for VQA Answer Classes
==========================================
This module creates the answer vocabulary for the VQA classification task.

VQA as Classification:
----------------------
Following the VQA v2 challenge approach, we treat VQA as a multi-class
classification problem over the top-K most frequent answers. This is because:
1. Open-ended answer generation is much harder and less reliable
2. Most VQA questions have common, repeated answers
3. Allows use of standard cross-entropy loss
4. Evaluation becomes straightforward accuracy

Answer Selection Strategy:
--------------------------
- Parse VQA v2 annotations
- Count frequency of each unique answer
- Select top-1000 most frequent answers
- Map each answer to a class index (0-999)
"""

import json
import os
from typing import Dict, List, Tuple, Optional
from collections import Counter
from pathlib import Path
import re


class AnswerVocabulary:
    """
    Vocabulary for VQA answer classification.
    
    This class handles:
    1. Loading VQA annotations
    2. Counting answer frequencies
    3. Building top-K answer vocabulary
    4. Answer-to-index mapping for classification
    
    Attributes:
        answer2idx (Dict[str, int]): Answer string to class index
        idx2answer (Dict[int, str]): Class index to answer string
        num_answers (int): Total number of answer classes
        answer_counts (Dict[str, int]): Frequency of each answer
    """
    
    def __init__(self, num_answers: int = 1000):
        """
        Initialize answer vocabulary.
        
        Args:
            num_answers: Number of top answers to keep (default: 1000)
        """
        self.num_answers = num_answers
        self.answer2idx: Dict[str, int] = {}
        self.idx2answer: Dict[int, str] = {}
        self.answer_counts: Dict[str, int] = {}
        self._is_built = False
    
    @staticmethod
    def preprocess_answer(answer: str) -> str:
        """
        Preprocess answer string for consistency.
        
        Following VQA v2 evaluation preprocessing:
        1. Convert to lowercase
        2. Remove articles (a, an, the)
        3. Remove punctuation
        4. Reduce multiple spaces
        
        Args:
            answer: Raw answer string
            
        Returns:
            Preprocessed answer
        
        Example:
            >>> AnswerVocabulary.preprocess_answer("The Blue car")
            "blue car"
        """
        answer = answer.lower()
        
        # Remove articles
        answer = re.sub(r'\b(a|an|the)\b', ' ', answer)
        
        # Remove punctuation
        answer = re.sub(r'[^\w\s]', '', answer)
        
        # Remove extra whitespace
        answer = re.sub(r'\s+', ' ', answer)
        
        return answer.strip()
    
    def build_from_annotations(
        self,
        annotations_path: str,
        save_path: Optional[str] = None
    ) -> None:
        """
        Build answer vocabulary from VQA v2 annotations file.
        
        VQA v2 annotation format:
        {
            "annotations": [
                {
                    "question_id": int,
                    "image_id": int,
                    "answers": [
                        {"answer": str, "answer_confidence": str, "answer_id": int},
                        ... (10 annotators)
                    ],
                    "multiple_choice_answer": str  # Most common answer
                },
                ...
            ]
        }
        
        Args:
            annotations_path: Path to VQA annotations JSON
            save_path: Optional path to save vocabulary
        """
        print(f"[AnswerVocab] Loading annotations from {annotations_path}")
        
        with open(annotations_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Count all answers (including from all annotators)
        answer_counter = Counter()
        
        for ann in data['annotations']:
            # Count the multiple_choice_answer (most common per question)
            primary_answer = self.preprocess_answer(ann['multiple_choice_answer'])
            answer_counter[primary_answer] += 1
            
            # Optionally also count individual annotator answers
            # This gives more robust frequency estimates
            for ans_dict in ann.get('answers', []):
                answer = self.preprocess_answer(ans_dict['answer'])
                answer_counter[answer] += 1
        
        # Store all counts for analysis
        self.answer_counts = dict(answer_counter)
        
        # Get top-K answers
        most_common = answer_counter.most_common(self.num_answers)
        
        print(f"[AnswerVocab] Total unique answers: {len(answer_counter)}")
        print(f"[AnswerVocab] Keeping top-{self.num_answers} answers")
        
        # Build mappings
        for idx, (answer, count) in enumerate(most_common):
            self.answer2idx[answer] = idx
            self.idx2answer[idx] = answer
        
        self._is_built = True
        
        # Print some statistics
        top_10 = most_common[:10]
        print(f"[AnswerVocab] Top 10 answers:")
        for answer, count in top_10:
            print(f"  {answer}: {count}")
        
        # Calculate coverage
        total_occurrences = sum(answer_counter.values())
        covered_occurrences = sum(count for _, count in most_common)
        coverage = covered_occurrences / total_occurrences * 100
        print(f"[AnswerVocab] Vocabulary coverage: {coverage:.2f}%")
        
        if save_path:
            self.save(save_path)
    
    def build_from_qa_pairs(
        self,
        qa_pairs: List[Dict],
        answer_key: str = "answer",
        save_path: Optional[str] = None
    ) -> None:
        """
        Build vocabulary from list of QA pairs.
        
        Alternative to annotation files, useful for custom datasets.
        
        Args:
            qa_pairs: List of dicts with answer field
            answer_key: Key to access answer in dict
            save_path: Optional path to save vocabulary
        """
        answer_counter = Counter()
        
        for qa in qa_pairs:
            answer = self.preprocess_answer(qa[answer_key])
            answer_counter[answer] += 1
        
        self.answer_counts = dict(answer_counter)
        most_common = answer_counter.most_common(self.num_answers)
        
        for idx, (answer, count) in enumerate(most_common):
            self.answer2idx[answer] = idx
            self.idx2answer[idx] = answer
        
        self._is_built = True
        
        if save_path:
            self.save(save_path)
    
    def encode(self, answer: str) -> int:
        """
        Encode answer string to class index.
        
        Args:
            answer: Answer string
            
        Returns:
            Class index, or -1 if answer not in vocabulary
        """
        preprocessed = self.preprocess_answer(answer)
        return self.answer2idx.get(preprocessed, -1)
    
    def decode(self, idx: int) -> str:
        """
        Decode class index to answer string.
        
        Args:
            idx: Class index
            
        Returns:
            Answer string, or "<UNKNOWN>" if index invalid
        """
        return self.idx2answer.get(idx, "<UNKNOWN>")
    
    def is_valid_answer(self, answer: str) -> bool:
        """
        Check if answer is in vocabulary.
        
        Args:
            answer: Answer string
            
        Returns:
            True if answer is in vocabulary
        """
        preprocessed = self.preprocess_answer(answer)
        return preprocessed in self.answer2idx
    
    def save(self, filepath: str) -> None:
        """
        Save vocabulary to JSON file.
        
        Args:
            filepath: Path to save vocabulary
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        vocab_data = {
            "num_answers": self.num_answers,
            "answer2idx": self.answer2idx,
            "answer_counts": self.answer_counts
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        
        print(f"[AnswerVocab] Saved vocabulary to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load vocabulary from JSON file.
        
        Args:
            filepath: Path to load vocabulary from
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.num_answers = vocab_data["num_answers"]
        self.answer2idx = vocab_data["answer2idx"]
        # Ensure integer keys
        self.idx2answer = {int(v): k for k, v in self.answer2idx.items()}
        self.answer_counts = vocab_data.get("answer_counts", {})
        self._is_built = True
        
        print(f"[AnswerVocab] Loaded vocabulary with {self.num_answers} answers")
    
    def get_answer_weights(self) -> List[float]:
        """
        Get inverse frequency weights for class balancing.
        
        Useful for weighted cross-entropy loss to handle
        class imbalance in answer distribution.
        
        Returns:
            List of weights for each answer class
        """
        if not self.answer_counts:
            return [1.0] * self.num_answers
        
        weights = []
        total = sum(self.answer_counts.get(self.idx2answer.get(i, ""), 1) 
                   for i in range(self.num_answers))
        
        for i in range(self.num_answers):
            answer = self.idx2answer.get(i, "")
            count = self.answer_counts.get(answer, 1)
            # Inverse frequency: less common answers get higher weight
            weight = total / (len(self.answer2idx) * count)
            weights.append(weight)
        
        return weights


def create_answer_vocabulary(
    annotations_path: str,
    num_answers: int = 1000,
    save_path: Optional[str] = None
) -> AnswerVocabulary:
    """
    Factory function to create answer vocabulary.
    
    Args:
        annotations_path: Path to VQA annotations
        num_answers: Number of answer classes
        save_path: Path to save vocabulary
        
    Returns:
        Built AnswerVocabulary instance
    """
    vocab = AnswerVocabulary(num_answers=num_answers)
    vocab.build_from_annotations(annotations_path, save_path)
    return vocab


if __name__ == "__main__":
    # Demo with sample data
    sample_qa_pairs = [
        {"answer": "yes"},
        {"answer": "no"},
        {"answer": "Yes"},
        {"answer": "blue"},
        {"answer": "red"},
        {"answer": "2"},
        {"answer": "two"},
        {"answer": "yes"},
        {"answer": "yes"},
        {"answer": "no"},
    ]
    
    vocab = AnswerVocabulary(num_answers=5)
    vocab.build_from_qa_pairs(sample_qa_pairs)
    
    print(f"\nVocabulary: {vocab.answer2idx}")
    print(f"\nEncode 'yes': {vocab.encode('yes')}")
    print(f"Encode 'YES': {vocab.encode('YES')}")  # Should match 'yes'
    print(f"Encode 'unknown': {vocab.encode('unknown')}")  # Should be -1
    print(f"Decode 0: {vocab.decode(0)}")
