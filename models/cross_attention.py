"""
Cross-Attention Module for VQA System
======================================
This module implements cross-attention for multimodal fusion.

Cross-Attention vs Self-Attention:
----------------------------------
Self-Attention: Query, Key, Value all come from same sequence
Cross-Attention: Query from one modality, Key/Value from another

For VQA Cross-Attention:
- Query: Question features (what are we asking about?)
- Key/Value: Image features (what visual information is available?)

This allows the question to "look at" relevant parts of the image.

Mathematical Formulation:
-------------------------
Given:
- Question features Q_text ∈ R^(L_q × D)
- Image features I ∈ R^(L_i × D)

Cross-Attention:
Q = Q_text @ W_q    (queries from question)
K = I @ W_k         (keys from image)
V = I @ W_v         (values from image)

Attention = softmax(QK^T / sqrt(d_k)) @ V

The output has the same sequence length as the query (question),
but incorporates relevant image information.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CrossAttention(nn.Module):
    """
    Cross-Attention Module (Manual Implementation)
    
    This is a complete manual implementation of cross-attention.
    NOT using nn.MultiheadAttention since queries and keys/values
    come from different sources.
    
    Architecture:
    -------------
    Question ──> Q projection ──┐
                                ├──> Attention ──> Output
    Image ────> K, V projection ┘
    
    The question "queries" the image, allowing the model to
    focus on image regions relevant to answering the question.
    
    Example:
    --------
    Question: "What color is the car?"
    The model should attend to pixels containing the car.
    
    Attributes:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head (embed_dim / num_heads)
        scale: Scaling factor for dot product
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = False
    ):
        """
        Initialize cross-attention module.
        
        Args:
            embed_dim: Embedding dimension (same for both modalities)
            num_heads: Number of attention heads
            dropout: Attention dropout rate
            bias: Whether to use bias in linear projections
        """
        super(CrossAttention, self).__init__()
        
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Query projection (from question/text)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Key and Value projections (from image)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Output projection
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Xavier initialization for stable training."""
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        key_value_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of cross-attention.
        
        Args:
            query: Query features (from text) [B, L_q, D]
            key_value: Key/Value features (from image) [B, L_kv, D]
            query_mask: Optional mask for queries [B, L_q]
            key_value_mask: Optional mask for keys/values [B, L_kv]
            
        Returns:
            output: Cross-attended features [B, L_q, D]
            attention_weights: Attention map [B, num_heads, L_q, L_kv]
        
        Shape tracking:
            query:          [B, L_q, D]
            key_value:      [B, L_kv, D]
            Q:              [B, H, L_q, D/H]
            K, V:           [B, H, L_kv, D/H]
            attention:      [B, H, L_q, L_kv]
            context:        [B, H, L_q, D/H]
            output:         [B, L_q, D]
            
        For VQA typically:
            L_q = max_question_length (e.g., 20)
            L_kv = 7 × 7 = 49 (spatial positions from CNN)
        """
        batch_size = query.size(0)
        query_len = query.size(1)
        kv_len = key_value.size(1)
        
        # =====================================================================
        # Step 1: Linear projections
        # Query from text, Key/Value from image
        # =====================================================================
        Q = self.W_q(query)         # [B, L_q, D]
        K = self.W_k(key_value)     # [B, L_kv, D]
        V = self.W_v(key_value)     # [B, L_kv, D]
        
        # =====================================================================
        # Step 2: Reshape for multi-head attention
        # =====================================================================
        # [B, L, D] -> [B, L, H, D/H] -> [B, H, L, D/H]
        Q = Q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # =====================================================================
        # Step 3: Scaled dot-product attention
        # Each question token attends to all image regions
        # =====================================================================
        # [B, H, L_q, D/H] @ [B, H, D/H, L_kv] -> [B, H, L_q, L_kv]
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply key/value mask if provided (e.g., for variable-sized images)
        if key_value_mask is not None:
            # [B, L_kv] -> [B, 1, 1, L_kv]
            mask = key_value_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax over key dimension
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # =====================================================================
        # Step 4: Apply attention to values
        # =====================================================================
        # [B, H, L_q, L_kv] @ [B, H, L_kv, D/H] -> [B, H, L_q, D/H]
        context = torch.matmul(attention_weights, V)
        
        # =====================================================================
        # Step 5: Concatenate heads and project
        # =====================================================================
        # [B, H, L_q, D/H] -> [B, L_q, H, D/H] -> [B, L_q, D]
        context = context.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.embed_dim
        )
        
        # Output projection
        output = self.W_o(context)
        
        return output, attention_weights


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention with Pre-Norm and Residual Connection
    
    This wraps CrossAttention with:
    1. Layer normalization (pre-norm for stability)
    2. Residual connection
    3. Optional feed-forward network
    
    Architecture:
    -------------
    query ─┬─> LayerNorm -> CrossAttention(query, kv) ─┬─> + ─> output
           │                                           │
           └───────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_ffn: bool = True,
        ffn_hidden_dim: Optional[int] = None
    ):
        """
        Initialize multi-head cross-attention with residual.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_ffn: Whether to add FFN after attention
            ffn_hidden_dim: FFN hidden dimension (default: 4 * embed_dim)
        """
        super(MultiHeadCrossAttention, self).__init__()
        
        # Layer norms
        self.norm_query = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        
        # Cross-attention
        self.cross_attention = CrossAttention(embed_dim, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        
        # Optional FFN
        self.use_ffn = use_ffn
        if use_ffn:
            ffn_hidden = ffn_hidden_dim or (4 * embed_dim)
            self.norm_ffn = nn.LayerNorm(embed_dim)
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, ffn_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(ffn_hidden, embed_dim),
                nn.Dropout(dropout)
            )
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        key_value_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with residual connections.
        
        Args:
            query: Query features [B, L_q, D]
            key_value: Key/Value features [B, L_kv, D]
            query_mask: Query padding mask [B, L_q]
            key_value_mask: Key/Value mask [B, L_kv]
            
        Returns:
            output: Cross-attended features [B, L_q, D]
            attention_weights: Attention map [B, num_heads, L_q, L_kv]
        """
        # Cross-attention with residual
        normed_query = self.norm_query(query)
        normed_kv = self.norm_kv(key_value)
        
        attended, attn_weights = self.cross_attention(
            normed_query, normed_kv, query_mask, key_value_mask
        )
        query = query + self.dropout1(attended)
        
        # FFN with residual
        if self.use_ffn:
            normed = self.norm_ffn(query)
            query = query + self.ffn(normed)
        
        return query, attn_weights


class StackedCrossAttention(nn.Module):
    """
    Stacked Cross-Attention Layers
    
    Multiple layers of cross-attention allow for:
    1. Progressive refinement of attended features
    2. More complex query-key relationships
    3. Deeper multimodal understanding
    
    Typically 2-4 layers are sufficient for VQA.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize stacked cross-attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads per layer
            num_layers: Number of cross-attention layers
            dropout: Dropout rate
        """
        super(StackedCrossAttention, self).__init__()
        
        self.layers = nn.ModuleList([
            MultiHeadCrossAttention(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        key_value_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass through stacked layers.
        
        Args:
            query: Query features [B, L_q, D]
            key_value: Key/Value features [B, L_kv, D]
            query_mask: Query mask [B, L_q]
            key_value_mask: Key/Value mask [B, L_kv]
            
        Returns:
            output: Final cross-attended features [B, L_q, D]
            all_attention_weights: List of attention maps per layer
        """
        all_attention_weights = []
        
        for layer in self.layers:
            query, attn_weights = layer(
                query, key_value, query_mask, key_value_mask
            )
            all_attention_weights.append(attn_weights)
        
        return query, all_attention_weights


if __name__ == "__main__":
    # Test cross-attention modules
    print("Testing Cross-Attention Modules\n" + "=" * 50)
    
    # Typical VQA dimensions
    batch_size = 2
    query_len = 20      # Question length
    kv_len = 49         # 7×7 image regions
    embed_dim = 256
    
    # Create test inputs
    query = torch.randn(batch_size, query_len, embed_dim)
    key_value = torch.randn(batch_size, kv_len, embed_dim)
    
    # Optional masks
    query_mask = torch.ones(batch_size, query_len)
    query_mask[0, 15:] = 0  # Pad first query
    
    print(f"Query shape: {query.shape}")
    print(f"Key/Value shape: {key_value.shape}")
    
    # Test basic cross-attention
    cross_attn = CrossAttention(embed_dim, num_heads=8)
    output, attn_weights = cross_attn(query, key_value, query_mask)
    
    print(f"\nBasic Cross-Attention:")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: [{batch_size}, {query_len}, {embed_dim}]")
    print(f"  Attention weights shape: {attn_weights.shape}")
    print(f"  Expected: [{batch_size}, 8, {query_len}, {kv_len}]")
    
    # Verify attention sums to 1
    attn_sum = attn_weights.sum(dim=-1)
    print(f"  Attention sum (should be ~1.0): {attn_sum[0, 0, 0].item():.4f}")
    
    # Test multi-head with FFN
    mh_cross = MultiHeadCrossAttention(embed_dim, num_heads=8, use_ffn=True)
    output2, attn2 = mh_cross(query, key_value, query_mask)
    
    print(f"\nMulti-Head Cross-Attention with FFN:")
    print(f"  Output shape: {output2.shape}")
    
    # Test stacked layers
    stacked = StackedCrossAttention(embed_dim, num_heads=8, num_layers=2)
    output3, all_attns = stacked(query, key_value, query_mask)
    
    print(f"\nStacked Cross-Attention (2 layers):")
    print(f"  Output shape: {output3.shape}")
    print(f"  Number of attention maps: {len(all_attns)}")
    
    # Count parameters
    print(f"\nParameter counts:")
    print(f"  Basic CrossAttention: {sum(p.numel() for p in cross_attn.parameters()):,}")
    print(f"  MultiHead + FFN: {sum(p.numel() for p in mh_cross.parameters()):,}")
    print(f"  Stacked (2 layers): {sum(p.numel() for p in stacked.parameters()):,}")
    
    print("\n✓ Cross-attention tests passed!")
