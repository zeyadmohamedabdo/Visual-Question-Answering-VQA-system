"""
Tokenizer Module for VQA System
================================
This module handles word-level tokenization for VQA questions.

Design Decision: Word-Level Tokenization
-----------------------------------------
We use word-level tokenization instead of subword (BPE) because:
1. VQA questions are short (avg ~6 words) and domain-limited
2. Improves interpretability for attention visualization
3. Keeps focus on multimodal attention rather than NLP complexity
4. Sufficient vocabulary coverage with ~10K words for VQA domain

This module provides:
- Vocabulary building from question corpus
- Text preprocessing (lowercase, punctuation)
- Token encoding/decoding with special tokens
"""

import re
import json
from typing import List, Dict, Optional, Tuple
from collections import Counter


# =============================================================================
# Special Token Definitions
# =============================================================================
# Why these tokens:
# - PAD: Enables batching of variable-length sequences
# - UNK: Handles out-of-vocabulary words gracefully
# - START/END: Mark sequence boundaries (useful for attention)

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
START_TOKEN = "<START>"
END_TOKEN = "<END>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN]
PAD_IDX = 0
UNK_IDX = 1
START_IDX = 2
END_IDX = 3


class Tokenizer:
    """
    Word-level tokenizer for VQA questions.
    
    This tokenizer:
    1. Preprocesses text (lowercase, remove punctuation except apostrophes)
    2. Splits on whitespace
    3. Maps words to integer indices
    4. Handles padding and truncation for batching
    
    Attributes:
        word2idx (Dict[str, int]): Mapping from word to index
        idx2word (Dict[int, str]): Mapping from index to word
        vocab_size (int): Total vocabulary size including special tokens
        max_length (int): Maximum sequence length for padding/truncation
    """
    
    def __init__(
        self,
        max_length: int = 20,
        vocab_size: Optional[int] = None
    ):
        """
        Initialize tokenizer.
        
        Args:
            max_length: Maximum sequence length (VQA questions are short)
            vocab_size: Optional limit on vocabulary size
        """
        self.max_length = max_length
        self.max_vocab_size = vocab_size
        
        # Initialize with special tokens only
        self.word2idx: Dict[str, int] = {
            PAD_TOKEN: PAD_IDX,
            UNK_TOKEN: UNK_IDX,
            START_TOKEN: START_IDX,
            END_TOKEN: END_IDX
        }
        self.idx2word: Dict[int, str] = {v: k for k, v in self.word2idx.items()}
        
        self._is_fitted = False
    
    @property
    def vocab_size(self) -> int:
        """Return total vocabulary size."""
        return len(self.word2idx)
    
    @staticmethod
    def preprocess(text: str) -> str:
        """
        Preprocess text for tokenization.
        
        Steps:
        1. Convert to lowercase (reduces vocabulary size)
        2. Replace punctuation with spaces (except apostrophes)
        3. Collapse multiple spaces
        
        Args:
            text: Raw input text
            
        Returns:
            Preprocessed text ready for tokenization
        
        Example:
            >>> Tokenizer.preprocess("What's in the IMAGE?")
            "what's in the image"
        """
        # Lowercase
        text = text.lower()
        
        # Replace punctuation except apostrophes and letters
        # Keep apostrophes for contractions like "what's", "don't"
        text = re.sub(r"[^\w\s']", " ", text)
        
        # Replace multiple spaces with single space
        text = re.sub(r"\s+", " ", text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize preprocessed text into words.
        
        Args:
            text: Raw text to tokenize
            
        Returns:
            List of word tokens
        """
        preprocessed = self.preprocess(text)
        tokens = preprocessed.split()
        return tokens
    
    def build_vocab(
        self,
        questions: List[str],
        min_freq: int = 2
    ) -> None:
        """
        Build vocabulary from a list of questions.
        
        This counts word frequencies and keeps the most frequent words
        up to max_vocab_size. Words appearing less than min_freq times
        are mapped to UNK.
        
        Args:
            questions: List of question strings
            min_freq: Minimum frequency to include word in vocab
        
        Side Effects:
            Updates self.word2idx and self.idx2word
        """
        # Count word frequencies across all questions
        word_counts = Counter()
        for question in questions:
            tokens = self.tokenize(question)
            word_counts.update(tokens)
        
        # Filter by minimum frequency
        filtered_words = [
            word for word, count in word_counts.items()
            if count >= min_freq
        ]
        
        # Sort by frequency (most common first)
        sorted_words = sorted(
            filtered_words,
            key=lambda w: word_counts[w],
            reverse=True
        )
        
        # Limit vocabulary size if specified
        if self.max_vocab_size is not None:
            # Account for special tokens
            max_words = self.max_vocab_size - len(SPECIAL_TOKENS)
            sorted_words = sorted_words[:max_words]
        
        # Build word2idx mapping (special tokens already added)
        current_idx = len(SPECIAL_TOKENS)
        for word in sorted_words:
            if word not in self.word2idx:
                self.word2idx[word] = current_idx
                self.idx2word[current_idx] = word
                current_idx += 1
        
        self._is_fitted = True
        print(f"[Tokenizer] Built vocabulary with {self.vocab_size} tokens")
        print(f"[Tokenizer] Filtered {len(word_counts) - len(sorted_words)} rare words")
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        padding: bool = True,
        truncation: bool = True
    ) -> Tuple[List[int], List[int]]:
        """
        Encode text to token indices.
        
        Args:
            text: Input text to encode
            add_special_tokens: If True, add START and END tokens
            padding: If True, pad to max_length
            truncation: If True, truncate to max_length
            
        Returns:
            Tuple of (token_ids, attention_mask)
            - token_ids: List of integer indices
            - attention_mask: 1 for real tokens, 0 for padding
            
        Shape:
            token_ids: [max_length] if padding else [seq_len]
            attention_mask: [max_length] if padding else [seq_len]
        """
        # Tokenize
        tokens = self.tokenize(text)
        
        # Add special tokens
        if add_special_tokens:
            tokens = [START_TOKEN] + tokens + [END_TOKEN]
        
        # Truncate if needed
        if truncation and len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            # Ensure END token is present after truncation
            if add_special_tokens:
                tokens[-1] = END_TOKEN
        
        # Convert to indices
        token_ids = [
            self.word2idx.get(token, UNK_IDX)
            for token in tokens
        ]
        
        # Create attention mask
        attention_mask = [1] * len(token_ids)
        
        # Pad if needed
        if padding and len(token_ids) < self.max_length:
            pad_length = self.max_length - len(token_ids)
            token_ids.extend([PAD_IDX] * pad_length)
            attention_mask.extend([0] * pad_length)
        
        return token_ids, attention_mask
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token indices back to text.
        
        Args:
            token_ids: List of token indices
            skip_special_tokens: If True, remove special tokens from output
            
        Returns:
            Decoded text string
        """
        tokens = []
        for idx in token_ids:
            word = self.idx2word.get(idx, UNK_TOKEN)
            if skip_special_tokens and word in SPECIAL_TOKENS:
                continue
            tokens.append(word)
        
        return " ".join(tokens)
    
    def save(self, filepath: str) -> None:
        """
        Save vocabulary to JSON file.
        
        Args:
            filepath: Path to save vocabulary
        """
        vocab_data = {
            "word2idx": self.word2idx,
            "max_length": self.max_length,
            "max_vocab_size": self.max_vocab_size
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        print(f"[Tokenizer] Saved vocabulary to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load vocabulary from JSON file.
        
        Args:
            filepath: Path to load vocabulary from
        """
        with open(filepath, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
        
        self.word2idx = vocab_data["word2idx"]
        # Ensure integer keys for idx2word
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.max_length = vocab_data.get("max_length", self.max_length)
        self.max_vocab_size = vocab_data.get("max_vocab_size", self.max_vocab_size)
        self._is_fitted = True
        print(f"[Tokenizer] Loaded vocabulary with {self.vocab_size} tokens")
    
    def batch_encode(
        self,
        texts: List[str],
        add_special_tokens: bool = True
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of input texts
            add_special_tokens: If True, add START and END tokens
            
        Returns:
            Tuple of (batch_token_ids, batch_attention_masks)
        """
        batch_ids = []
        batch_masks = []
        
        for text in texts:
            ids, mask = self.encode(text, add_special_tokens=add_special_tokens)
            batch_ids.append(ids)
            batch_masks.append(mask)
        
        return batch_ids, batch_masks


# =============================================================================
# Convenience Functions
# =============================================================================

def create_tokenizer_from_questions(
    questions: List[str],
    max_length: int = 20,
    vocab_size: Optional[int] = 10000,
    min_freq: int = 2,
    save_path: Optional[str] = None
) -> Tokenizer:
    """
    Factory function to create and fit a tokenizer.
    
    Args:
        questions: List of question strings
        max_length: Maximum sequence length
        vocab_size: Maximum vocabulary size
        min_freq: Minimum word frequency
        save_path: Optional path to save vocabulary
        
    Returns:
        Fitted Tokenizer instance
    """
    tokenizer = Tokenizer(max_length=max_length, vocab_size=vocab_size)
    tokenizer.build_vocab(questions, min_freq=min_freq)
    
    if save_path:
        tokenizer.save(save_path)
    
    return tokenizer


if __name__ == "__main__":
    # Demo usage
    sample_questions = [
        "What color is the cat?",
        "How many people are there?",
        "Is this a beach?",
        "What is the man doing?",
        "What's in the background?"
    ]
    
    tokenizer = Tokenizer(max_length=15, vocab_size=1000)
    tokenizer.build_vocab(sample_questions, min_freq=1)
    
    # Test encoding
    test_question = "What color is the dog?"
    token_ids, attention_mask = tokenizer.encode(test_question)
    
    print(f"\nTest Question: {test_question}")
    print(f"Token IDs: {token_ids}")
    print(f"Attention Mask: {attention_mask}")
    print(f"Decoded: {tokenizer.decode(token_ids)}")
