"""
Multimodal Fusion Module for VQA System
========================================
This module combines image and text features for VQA.

Fusion Strategy:
---------------
We use cross-attention based fusion where:
1. Image features are reshaped to sequence format
2. Question features query relevant image regions
3. Attended features are combined via gating
4. Final fused representation captures question-relevant visual info

Why Cross-Attention Fusion (not simple concatenation):
-----------------------------------------------------
1. Selective attention: Only relevant image regions contribute
2. Interpretable: Attention maps show which regions matter
3. Question-conditional: Same image, different questions → different focus
4. Efficient: Compresses high-dim image into question-sized representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

from models.cross_attention import StackedCrossAttention, MultiHeadCrossAttention


class ImageFeatureProjector(nn.Module):
    """
    Project and reshape CNN features for cross-attention.
    
    CNN outputs: [B, C, H, W] (e.g., [B, 512, 7, 7])
    Cross-attention needs: [B, H*W, D] (e.g., [B, 49, 256])
    
    This module:
    1. Reshapes spatial dims to sequence
    2. Projects channel dim to embed_dim
    3. Adds optional learnable position embeddings
    """
    
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        spatial_size: int = 7,
        use_position_embed: bool = True,
        dropout: float = 0.1
    ):
        """
        Initialize image feature projector.
        
        Args:
            in_channels: CNN output channels (e.g., 512)
            embed_dim: Target embedding dimension (e.g., 256)
            spatial_size: Spatial dimension of CNN output (e.g., 7)
            use_position_embed: Whether to add position embeddings
            dropout: Dropout rate
        """
        super(ImageFeatureProjector, self).__init__()
        
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.spatial_size = spatial_size
        self.num_positions = spatial_size * spatial_size  # 49 for 7×7
        
        # Channel projection
        self.projection = nn.Sequential(
            nn.Linear(in_channels, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )
        
        # Learnable position embeddings for spatial locations
        self.use_position_embed = use_position_embed
        if use_position_embed:
            self.position_embedding = nn.Parameter(
                torch.randn(1, self.num_positions, embed_dim) * 0.02
            )
    
    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Project and reshape image features.
        
        Args:
            image_features: CNN output [B, C, H, W]
            
        Returns:
            projected: Projected features [B, H*W, embed_dim]
            
        Shape tracking:
            Input:      [B, 512, 7, 7]
            Reshape:    [B, 512, 49] -> [B, 49, 512]
            Project:    [B, 49, 256]
            + pos_emb:  [B, 49, 256]
        """
        batch_size, channels, height, width = image_features.shape
        num_positions = height * width
        
        # Reshape: [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
        x = image_features.view(batch_size, channels, num_positions)
        x = x.permute(0, 2, 1)  # [B, H*W, C]
        
        # Project to embed_dim
        x = self.projection(x)  # [B, H*W, embed_dim]
        
        # Add position embeddings
        if self.use_position_embed:
            x = x + self.position_embedding[:, :num_positions, :]
        
        return x


class GatingMechanism(nn.Module):
    """
    Gating mechanism for controlled feature fusion.
    
    Gating allows the model to learn how much of each modality
    to incorporate in the final representation.
    
    g = sigmoid(W_g @ [x; y])
    output = g * x + (1 - g) * y
    
    This is softer than hard attention and allows gradients
    to flow through both pathways.
    """
    
    def __init__(self, embed_dim: int):
        """
        Initialize gating mechanism.
        
        Args:
            embed_dim: Feature dimension
        """
        super(GatingMechanism, self).__init__()
        
        # Gate computation from concatenated features
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply gating to fuse two feature sets.
        
        Args:
            x: First feature set [B, L, D] or [B, D]
            y: Second feature set [B, L, D] or [B, D]
            
        Returns:
            Gated fusion [B, L, D] or [B, D]
        """
        # Concatenate for gate computation
        concat = torch.cat([x, y], dim=-1)
        
        # Compute gate values
        g = self.gate(concat)
        
        # Gated fusion
        output = g * x + (1 - g) * y
        
        return output


class MultimodalFusion(nn.Module):
    """
    Complete Multimodal Fusion Module for VQA
    
    This module combines image and question features using:
    1. Image feature projection and position encoding
    2. Cross-attention (question attends to image)
    3. Feature aggregation (pooling over sequence)
    4. Gated fusion of attended and original features
    
    Architecture:
    -------------
    Image Features [B, 512, 7, 7]
         ↓
    Project & Reshape [B, 49, 256]
         ↓
    ┌────────────────────────────────┐
    │  Cross-Attention               │
    │  Query: Question [B, L, 256]   │←── Question Features
    │  Key/Value: Image [B, 49, 256] │
    │  Output: Attended [B, L, 256]  │
    └────────────────────────────────┘
         ↓
    Pool (mean over sequence)
         ↓
    Gated Fusion with Question
         ↓
    Fused Features [B, 256]
    """
    
    def __init__(
        self,
        image_channels: int = 512,
        image_spatial_size: int = 7,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_cross_layers: int = 2,
        dropout: float = 0.1,
        use_gating: bool = True
    ):
        """
        Initialize multimodal fusion.
        
        Args:
            image_channels: CNN output channels
            image_spatial_size: CNN spatial size (7 for standard ResNet)
            embed_dim: Embedding dimension for fusion
            num_heads: Cross-attention heads
            num_cross_layers: Number of cross-attention layers
            dropout: Dropout rate
            use_gating: Whether to use gating mechanism
        """
        super(MultimodalFusion, self).__init__()
        
        self.embed_dim = embed_dim
        self.use_gating = use_gating
        
        # Project image features to embed_dim
        self.image_projector = ImageFeatureProjector(
            in_channels=image_channels,
            embed_dim=embed_dim,
            spatial_size=image_spatial_size,
            use_position_embed=True,
            dropout=dropout
        )
        
        # Cross-attention layers
        self.cross_attention = StackedCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_cross_layers,
            dropout=dropout
        )
        
        # Gating for final fusion
        if use_gating:
            self.gate = GatingMechanism(embed_dim)
        
        # Layer norm for final output
        self.output_norm = nn.LayerNorm(embed_dim)
        
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Fuse image and text features.
        
        Args:
            image_features: CNN output [B, C, H, W]
            text_features: Encoded question [B, L_text, D]
            text_mask: Question padding mask [B, L_text]
            
        Returns:
            fused: Fused multimodal features [B, D]
            aux_outputs: Dictionary with attention maps for visualization
            
        Shape tracking:
            image_features:  [B, 512, 7, 7]
            image_projected: [B, 49, 256]
            text_features:   [B, 20, 256]
            cross_attended:  [B, 20, 256]
            pooled:          [B, 256]
            fused:           [B, 256]
        """
        batch_size = image_features.size(0)
        
        # =====================================================================
        # Step 1: Project image features to embed_dim sequence
        # =====================================================================
        # [B, 512, 7, 7] -> [B, 49, 256]
        image_projected = self.image_projector(image_features)
        
        # =====================================================================
        # Step 2: Cross-attention (question attends to image)
        # =====================================================================
        # Query: text [B, L_text, D]
        # Key/Value: image [B, 49, D]
        # Output: [B, L_text, D]
        cross_attended, attention_weights = self.cross_attention(
            query=text_features,
            key_value=image_projected,
            query_mask=text_mask,
            key_value_mask=None  # All image positions valid
        )
        
        # =====================================================================
        # Step 3: Pool attended features
        # =====================================================================
        # Mean pooling over text sequence (masked)
        if text_mask is not None:
            mask = text_mask.unsqueeze(-1).float()  # [B, L, 1]
            attended_pooled = (cross_attended * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            attended_pooled = cross_attended.mean(dim=1)  # [B, D]
        
        # Also pool original text features for fusion
        if text_mask is not None:
            text_pooled = (text_features * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            text_pooled = text_features.mean(dim=1)
        
        # =====================================================================
        # Step 4: Gated fusion
        # =====================================================================
        if self.use_gating:
            # Combine attended (image-informed) and original text features
            fused = self.gate(attended_pooled, text_pooled)
        else:
            # Simple addition
            fused = attended_pooled + text_pooled
        
        # Final normalization
        fused = self.output_norm(fused)
        
        # Auxiliary outputs for visualization/debugging
        aux_outputs = {
            'cross_attention_weights': attention_weights,
            'image_projected': image_projected,
            'attended_pooled': attended_pooled,
            'text_pooled': text_pooled
        }
        
        return fused, aux_outputs
    
    def get_attention_visualization(
        self,
        attention_weights: list,
        spatial_size: int = 7
    ) -> torch.Tensor:
        """
        Reshape attention weights for visualization.
        
        Args:
            attention_weights: List of [B, H, L_q, L_kv] tensors
            spatial_size: Original spatial size (e.g., 7)
            
        Returns:
            Attention maps [B, L_q, H, W] averaged over heads and layers
        """
        # Average over layers
        avg_attn = torch.stack(attention_weights, dim=0).mean(dim=0)
        
        # Average over heads: [B, H, L_q, L_kv] -> [B, L_q, L_kv]
        avg_attn = avg_attn.mean(dim=1)
        
        # Reshape to spatial: [B, L_q, H, W]
        batch_size, query_len, kv_len = avg_attn.shape
        attn_map = avg_attn.view(batch_size, query_len, spatial_size, spatial_size)
        
        return attn_map


if __name__ == "__main__":
    # Test fusion module
    print("Testing Multimodal Fusion Module\n" + "=" * 50)
    
    # Typical VQA dimensions
    batch_size = 2
    image_channels = 512
    image_spatial = 7
    text_length = 20
    embed_dim = 256
    
    # Create test inputs
    image_features = torch.randn(batch_size, image_channels, image_spatial, image_spatial)
    text_features = torch.randn(batch_size, text_length, embed_dim)
    text_mask = torch.ones(batch_size, text_length)
    text_mask[0, 15:] = 0  # Pad first sequence
    
    print(f"Image features shape: {image_features.shape}")
    print(f"Text features shape: {text_features.shape}")
    
    # Create fusion module
    fusion = MultimodalFusion(
        image_channels=image_channels,
        image_spatial_size=image_spatial,
        embed_dim=embed_dim,
        num_heads=8,
        num_cross_layers=2,
        use_gating=True
    )
    
    # Forward pass
    with torch.no_grad():
        fused, aux = fusion(image_features, text_features, text_mask)
    
    print(f"\nFused features shape: {fused.shape}")
    print(f"Expected: [{batch_size}, {embed_dim}]")
    
    # Check auxiliary outputs
    print(f"\nAuxiliary outputs:")
    print(f"  Cross-attention layers: {len(aux['cross_attention_weights'])}")
    print(f"  Attention weight shape: {aux['cross_attention_weights'][0].shape}")
    print(f"  Image projected shape: {aux['image_projected'].shape}")
    
    # Test attention visualization
    attn_vis = fusion.get_attention_visualization(
        aux['cross_attention_weights'], image_spatial
    )
    print(f"  Attention visualization shape: {attn_vis.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in fusion.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Verify output shape
    assert fused.shape == (batch_size, embed_dim), "Fused shape mismatch!"
    
    print("\n✓ Multimodal fusion tests passed!")
