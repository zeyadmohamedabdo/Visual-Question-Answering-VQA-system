"""
Transformer Text Encoder for VQA System
========================================
This module implements a custom Transformer encoder for question encoding.

Design Philosophy:
-----------------
We build the Transformer from scratch (not using pretrained BERT weights)
to satisfy academic requirements. The architecture follows the original
"Attention Is All You Need" paper with customizations for VQA.

Key Components:
--------------
1. Token Embedding: Learnable word embeddings
2. Positional Encoding: Sinusoidal position information
3. Multi-Head Self-Attention: Implemented manually
4. Feed-Forward Networks: Position-wise transformations
5. Layer Normalization: Pre-norm variant for stability

Output:
-------
Sequence of contextualized token representations [B, seq_len, embed_dim]
The output preserves sequential structure for cross-attention with image features.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding
    
    Why Positional Encoding:
    -----------------------
    Transformers have no inherent notion of position (unlike RNNs).
    Positional encoding injects sequence order information.
    
    Mathematical Formulation:
    -------------------------
    For position pos and dimension i:
    
    PE(pos, 2i)     = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i + 1) = cos(pos / 10000^(2i/d_model))
    
    Properties:
    - Different frequencies for different dimensions
    - PE(pos+k) can be represented as linear function of PE(pos)
    - Extrapolates to longer sequences than seen during training
    
    Why Sinusoidal (not learned):
    ----------------------------
    1. Generalizes to longer sequences
    2. No additional parameters
    3. Proven effective for NLP tasks
    """
    
    def __init__(self, embed_dim: int, max_length: int = 512, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            embed_dim: Embedding dimension (d_model)
            max_length: Maximum sequence length to precompute
            dropout: Dropout rate
        """
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Precompute positional encodings
        # Shape: [max_length, embed_dim]
        pe = torch.zeros(max_length, embed_dim)
        
        # Position indices: [max_length, 1]
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        # Dimension indices for sin/cos alternation
        # div_term: [embed_dim/2]
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * 
            (-math.log(10000.0) / embed_dim)
        )
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions
        
        # Add batch dimension: [1, max_length, embed_dim]
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Token embeddings [B, seq_len, embed_dim]
            
        Returns:
            Position-encoded embeddings [B, seq_len, embed_dim]
        """
        seq_len = x.size(1)
        
        # Add positional encoding (broadcast over batch)
        # x: [B, seq_len, embed_dim] + pe: [1, seq_len, embed_dim]
        x = x + self.pe[:, :seq_len, :]
        
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (Manual Implementation)
    
    This is a complete manual implementation of multi-head attention,
    NOT using nn.MultiheadAttention, to satisfy academic requirements.
    
    Mathematical Formulation:
    -------------------------
    For each head h:
    
    Q_h = X @ W_q^h, K_h = X @ W_k^h, V_h = X @ W_v^h
    
    Attention_h = softmax(Q_h @ K_h^T / sqrt(d_k)) @ V_h
    
    MultiHead = Concat(head_1, ..., head_H) @ W_o
    
    Where d_k = embed_dim / num_heads
    
    Why Multi-Head:
    --------------
    1. Attend to different representation subspaces
    2. Different heads can focus on different aspects
       (e.g., syntax vs semantics)
    3. Increases model capacity without proportional compute
    
    Attention Intuition:
    -------------------
    - Query: "What am I looking for?"
    - Key: "What do I contain?"
    - Value: "What information do I provide?"
    
    Attention score = how well query matches key
    Output = weighted sum of values by attention scores
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        """
        Initialize multi-head self-attention.
        
        Args:
            embed_dim: Total embedding dimension (d_model)
            num_heads: Number of attention heads (H)
            dropout: Attention dropout rate
        """
        super(MultiHeadSelfAttention, self).__init__()
        
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # d_k = d_v
        
        # Scaling factor for dot product attention
        self.scale = math.sqrt(self.head_dim)
        
        # Linear projections for Q, K, V
        # Could use single linear for efficiency, but separate for clarity
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of multi-head self-attention.
        
        Args:
            x: Input sequence [B, seq_len, embed_dim]
            attention_mask: Optional mask [B, seq_len]
                           1 = attend, 0 = ignore (padding)
        
        Returns:
            output: Attended output [B, seq_len, embed_dim]
            attention_weights: Attention weights [B, num_heads, seq_len, seq_len]
        
        Shape tracking:
            Input x:     [B, L, D]
            Q, K, V:     [B, L, D]
            Reshape:     [B, L, H, D/H] -> [B, H, L, D/H]
            Attention:   [B, H, L, L]
            Output:      [B, H, L, D/H] -> [B, L, D]
        """
        batch_size, seq_len, _ = x.shape
        
        # =====================================================================
        # Step 1: Linear projections to get Q, K, V
        # =====================================================================
        Q = self.W_q(x)  # [B, L, D]
        K = self.W_k(x)  # [B, L, D]
        V = self.W_v(x)  # [B, L, D]
        
        # =====================================================================
        # Step 2: Reshape for multi-head attention
        # Split last dimension into (num_heads, head_dim)
        # =====================================================================
        # [B, L, D] -> [B, L, H, D/H] -> [B, H, L, D/H]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # =====================================================================
        # Step 3: Compute scaled dot-product attention
        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
        # =====================================================================
        # [B, H, L, D/H] @ [B, H, D/H, L] -> [B, H, L, L]
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply attention mask (for padding)
        if attention_mask is not None:
            # attention_mask: [B, L] -> [B, 1, 1, L] for broadcasting
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Replace masked positions with large negative value
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax over last dimension (keys)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # =====================================================================
        # Step 4: Apply attention to values
        # =====================================================================
        # [B, H, L, L] @ [B, H, L, D/H] -> [B, H, L, D/H]
        context = torch.matmul(attention_weights, V)
        
        # =====================================================================
        # Step 5: Concatenate heads and project
        # =====================================================================
        # [B, H, L, D/H] -> [B, L, H, D/H] -> [B, L, D]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        output = self.W_o(context)
        
        return output, attention_weights


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    Architecture:
    -------------
    FFN(x) = ReLU(x @ W_1 + b_1) @ W_2 + b_2
    
    Typically: d_ff = 4 * d_model (expansion then compression)
    
    Why FFN:
    -------
    1. Non-linear transformation per position
    2. Increases model capacity
    3. Processes each position independently
    
    The transformer alternates between:
    - Self-attention: mixing information across positions
    - FFN: processing information at each position
    """
    
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.1
    ):
        """
        Initialize feed-forward network.
        
        Args:
            embed_dim: Input/output dimension (d_model)
            hidden_dim: Hidden layer dimension (d_ff), typically 4 * embed_dim
            dropout: Dropout rate
        """
        super(FeedForwardNetwork, self).__init__()
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [B, seq_len, embed_dim]
            
        Returns:
            Output [B, seq_len, embed_dim]
        """
        # [B, L, D] -> [B, L, D_ff] -> [B, L, D]
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer
    
    Architecture (Pre-Norm variant):
    ---------------------------------
    x ─┬─> LayerNorm -> Self-Attention ─┬─> + ─┬─> LayerNorm -> FFN ─┬─> + -> out
       │                                │      │                     │
       └────────────────────────────────┘      └─────────────────────┘
    
    Pre-Norm vs Post-Norm:
    ----------------------
    Pre-Norm: LayerNorm before attention/FFN (used here)
    Post-Norm: LayerNorm after residual (original paper)
    
    Pre-Norm is more stable for training deep networks.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        dropout: float = 0.1
    ):
        """
        Initialize encoder layer.
        
        Args:
            embed_dim: Embedding dimension (d_model)
            num_heads: Number of attention heads
            ffn_hidden_dim: FFN hidden dimension
            dropout: Dropout rate
        """
        super(TransformerEncoderLayer, self).__init__()
        
        # Self-attention block
        self.self_attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feed-forward block
        self.ffn = FeedForwardNetwork(embed_dim, ffn_hidden_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder layer.
        
        Args:
            x: Input [B, seq_len, embed_dim]
            attention_mask: Optional padding mask [B, seq_len]
            
        Returns:
            output: Encoded output [B, seq_len, embed_dim]
            attention_weights: Self-attention weights [B, num_heads, seq_len, seq_len]
        """
        # Self-attention with residual (pre-norm)
        normed = self.norm1(x)
        attended, attn_weights = self.self_attention(normed, attention_mask)
        x = x + self.dropout1(attended)
        
        # FFN with residual (pre-norm)
        normed = self.norm2(x)
        ff_out = self.ffn(normed)
        x = x + self.dropout2(ff_out)
        
        return x, attn_weights


class TransformerTextEncoder(nn.Module):
    """
    Complete Transformer Encoder for Question Encoding
    
    Architecture:
    -------------
    Input: Token indices [B, seq_len]
    
    Token Embedding:    [B, seq_len] -> [B, seq_len, embed_dim]
    Positional Encoding: Add position information
    Encoder Layers:     N × TransformerEncoderLayer
    
    Output: [B, seq_len, embed_dim] - contextualized representations
    
    Design Choices for VQA:
    ----------------------
    1. Relatively small (4 layers, 256 dim) - questions are short
    2. Output preserves sequence - needed for cross-attention
    3. No decoder - this is encoding only
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_hidden_dim: int = 1024,
        max_length: int = 50,
        dropout: float = 0.1,
        pad_idx: int = 0
    ):
        """
        Initialize Transformer text encoder.
        
        Args:
            vocab_size: Vocabulary size
            embed_dim: Embedding dimension (d_model)
            num_layers: Number of encoder layers
            num_heads: Number of attention heads
            ffn_hidden_dim: FFN hidden dimension
            max_length: Maximum sequence length
            dropout: Dropout rate
            pad_idx: Padding token index for embedding
        """
        super(TransformerTextEncoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.pad_idx = pad_idx
        
        # Token embedding
        self.token_embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim, max_length, dropout)
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ffn_hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # Initialize embeddings
        self._init_embeddings()
        
    def _init_embeddings(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.token_embedding.weight, mean=0, std=self.embed_dim ** -0.5)
        # Zero out padding embedding
        if self.pad_idx is not None:
            self.token_embedding.weight.data[self.pad_idx].zero_()
    
    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode question tokens.
        
        Args:
            token_ids: Token indices [B, seq_len]
            attention_mask: Padding mask [B, seq_len], 1=real, 0=pad
            
        Returns:
            encoded: Contextualized representations [B, seq_len, embed_dim]
            pooled: Pooled question representation [B, embed_dim]
            
        Shape tracking:
            token_ids:       [B, L]
            After embedding: [B, L, D]
            After pos enc:   [B, L, D]
            After layers:    [B, L, D]
            pooled:          [B, D] (mean pooling)
        """
        # Token embedding
        # [B, L] -> [B, L, D]
        x = self.token_embedding(token_ids)
        
        # Scale embeddings (as in original transformer)
        x = x * math.sqrt(self.embed_dim)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through encoder layers
        all_attention_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, attention_mask)
            all_attention_weights.append(attn_weights)
        
        # Final normalization
        encoded = self.final_norm(x)
        
        # Compute pooled representation (for potential uses)
        # Masked mean pooling
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
            pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = encoded.mean(dim=1)
        
        return encoded, pooled
    
    def get_attention_weights(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> list:
        """
        Get attention weights from all layers (for visualization).
        
        Args:
            token_ids: Token indices [B, seq_len]
            attention_mask: Padding mask [B, seq_len]
            
        Returns:
            List of attention weights per layer
            Each: [B, num_heads, seq_len, seq_len]
        """
        x = self.token_embedding(token_ids) * math.sqrt(self.embed_dim)
        x = self.positional_encoding(x)
        
        attention_weights = []
        for layer in self.layers:
            x, attn = layer(x, attention_mask)
            attention_weights.append(attn)
        
        return attention_weights


if __name__ == "__main__":
    # Test Transformer text encoder
    print("Testing Transformer Text Encoder\n" + "=" * 50)
    
    # Create encoder
    encoder = TransformerTextEncoder(
        vocab_size=10000,
        embed_dim=256,
        num_layers=4,
        num_heads=8,
        ffn_hidden_dim=1024,
        max_length=50
    )
    
    # Test input: batch of token sequences
    batch_size = 2
    seq_len = 20
    token_ids = torch.randint(0, 10000, (batch_size, seq_len))
    
    # Create attention mask (simulate padding)
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[0, 15:] = 0  # First sequence padded from position 15
    attention_mask[1, 18:] = 0  # Second sequence padded from position 18
    
    print(f"Token IDs shape: {token_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    
    # Forward pass
    with torch.no_grad():
        encoded, pooled = encoder(token_ids, attention_mask)
    
    print(f"\nEncoded output shape: {encoded.shape}")
    print(f"Expected: [{batch_size}, {seq_len}, 256]")
    print(f"Pooled output shape: {pooled.shape}")
    print(f"Expected: [{batch_size}, 256]")
    
    # Verify shapes
    assert encoded.shape == (batch_size, seq_len, 256), "Encoded shape mismatch!"
    assert pooled.shape == (batch_size, 256), "Pooled shape mismatch!"
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Test attention visualization
    attn_weights = encoder.get_attention_weights(token_ids, attention_mask)
    print(f"\nNumber of attention layers: {len(attn_weights)}")
    print(f"Attention shape per layer: {attn_weights[0].shape}")
    
    print("\n✓ Transformer text encoder tests passed!")
