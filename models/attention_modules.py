"""
Attention Modules for VQA System
=================================
This module implements custom attention mechanisms for the CNN backbone.

Implemented Attention Mechanisms:
1. SE (Squeeze-and-Excitation) Attention - Channel attention
2. Spatial Attention - Location-based attention
3. CBAM-style combined attention (optional)

These modules are designed to be inserted between CNN blocks to enable
the network to focus on relevant features for VQA tasks.

Mathematical Background:
-----------------------
Attention mechanisms learn to weight features by their importance.
Instead of treating all channels/locations equally, attention allows
the network to emphasize task-relevant information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SEAttention(nn.Module):
    """
    Squeeze-and-Excitation (SE) Attention Block
    
    Reference: "Squeeze-and-Excitation Networks" (Hu et al., 2018)
    
    SE attention performs channel-wise recalibration by:
    1. SQUEEZE: Global average pooling to get channel statistics
    2. EXCITATION: FC layers to learn channel interdependencies
    3. SCALE: Multiply original features by learned channel weights
    
    Mathematical Formulation:
    -------------------------
    Given input X ∈ R^(C×H×W):
    
    1. Squeeze: z_c = (1/H×W) Σ_i Σ_j X_c(i,j)  → z ∈ R^C
       This captures global spatial information per channel.
    
    2. Excitation: s = σ(W_2 · ReLU(W_1 · z))  → s ∈ R^C
       W_1 ∈ R^(C/r × C), W_2 ∈ R^(C × C/r), r = reduction ratio
       This models channel interdependencies.
    
    3. Scale: X' = s ⊗ X  (channel-wise multiplication)
       Each channel is weighted by its learned importance.
    
    Why SE for VQA:
    --------------
    In VQA, certain feature channels may be more relevant to the question.
    For example, color channels for "What color..." questions,
    or shape channels for "How many..." questions.
    SE allows the network to dynamically emphasize relevant channels.
    
    Attributes:
        fc1: First fully connected layer (squeeze)
        fc2: Second fully connected layer (excitation)
        reduction: Reduction ratio for bottleneck
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        """
        Initialize SE attention block.
        
        Args:
            channels: Number of input channels (C)
            reduction: Reduction ratio for bottleneck (r)
                       Higher = fewer parameters, less capacity
                       Typical values: 8, 16, 32
        """
        super(SEAttention, self).__init__()
        
        # Ensure reduced dimension is at least 1
        reduced_channels = max(channels // reduction, 1)
        
        # Squeeze: Global Average Pooling
        # Implemented in forward() using adaptive_avg_pool2d
        
        # Excitation: Two FC layers with ReLU and Sigmoid
        self.fc1 = nn.Linear(channels, reduced_channels, bias=False)
        self.fc2 = nn.Linear(reduced_channels, channels, bias=False)
        
        # Store for shape annotations
        self.channels = channels
        self.reduced_channels = reduced_channels
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SE attention.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Reweighted tensor of shape [B, C, H, W]
            
        Shape tracking:
            Input:       [B, C, H, W]
            After GAP:   [B, C, 1, 1] -> [B, C]
            After FC1:   [B, C/r]
            After FC2:   [B, C]
            After Sigmoid: [B, C] -> [B, C, 1, 1]
            Output:      [B, C, H, W] (element-wise multiply)
        """
        batch_size, channels, height, width = x.shape
        
        # =====================================================================
        # SQUEEZE: Global Average Pooling
        # Aggregates spatial information into channel descriptor
        # =====================================================================
        # [B, C, H, W] -> [B, C, 1, 1] -> [B, C]
        squeezed = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        
        # =====================================================================
        # EXCITATION: Learn channel interdependencies
        # Two FC layers form a bottleneck to limit parameters
        # =====================================================================
        # [B, C] -> [B, C/r]
        excited = F.relu(self.fc1(squeezed), inplace=True)
        
        # [B, C/r] -> [B, C]
        excited = torch.sigmoid(self.fc2(excited))
        
        # =====================================================================
        # SCALE: Reweight channels
        # Reshape for broadcasting and multiply
        # =====================================================================
        # [B, C] -> [B, C, 1, 1] for broadcasting
        scale = excited.view(batch_size, channels, 1, 1)
        
        # [B, C, H, W] * [B, C, 1, 1] -> [B, C, H, W]
        return x * scale


class SpatialAttention(nn.Module):
    """
    Spatial Attention Block
    
    Reference: Inspired by CBAM (Woo et al., 2018)
    
    Spatial attention focuses on WHERE in the image to look.
    It generates a spatial attention map that weights different locations.
    
    Mathematical Formulation:
    -------------------------
    Given input X ∈ R^(C×H×W):
    
    1. Channel pooling: Compute max and avg across channels
       F_max = max_c(X)  → F_max ∈ R^(1×H×W)
       F_avg = avg_c(X)  → F_avg ∈ R^(1×H×W)
    
    2. Concatenate: F = [F_max ; F_avg]  → F ∈ R^(2×H×W)
    
    3. Convolution: M = σ(Conv(F))  → M ∈ R^(1×H×W)
       Where Conv is a 7×7 convolution
    
    4. Scale: X' = M ⊗ X  (spatial-wise multiplication)
    
    Why Spatial Attention for VQA:
    -----------------------------
    VQA requires focusing on specific image regions.
    For "What is the person holding?", the model should attend
    to the person and their hands, not the background.
    Spatial attention learns to highlight relevant regions.
    
    Attributes:
        conv: Convolutional layer for attention map
        kernel_size: Size of convolution kernel
    """
    
    def __init__(self, kernel_size: int = 7):
        """
        Initialize spatial attention block.
        
        Args:
            kernel_size: Size of convolution kernel (typically 7)
                        Larger = larger receptive field for attention
        """
        super(SpatialAttention, self).__init__()
        
        # Ensure kernel size is odd for 'same' padding
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        
        # Conv layer: 2 input channels (max + avg) -> 1 output channel
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of spatial attention.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Spatially-weighted tensor of shape [B, C, H, W]
            
        Shape tracking:
            Input:       [B, C, H, W]
            Max pool:    [B, 1, H, W]
            Avg pool:    [B, 1, H, W]
            Concat:      [B, 2, H, W]
            After Conv:  [B, 1, H, W]
            Output:      [B, C, H, W]
        """
        # =====================================================================
        # CHANNEL POOLING: Aggregate channel information
        # Both max and avg provide complementary information
        # Max: highlights strongest features
        # Avg: provides overall feature strength
        # =====================================================================
        # [B, C, H, W] -> [B, 1, H, W]
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        
        # =====================================================================
        # CONCATENATE: Combine pooled features
        # =====================================================================
        # [B, 1, H, W] + [B, 1, H, W] -> [B, 2, H, W]
        pooled = torch.cat([max_pool, avg_pool], dim=1)
        
        # =====================================================================
        # CONVOLUTION: Generate attention map
        # 7×7 kernel captures local context for attention
        # =====================================================================
        # [B, 2, H, W] -> [B, 1, H, W]
        attention_map = torch.sigmoid(self.conv(pooled))
        
        # =====================================================================
        # SCALE: Apply spatial attention
        # =====================================================================
        # [B, C, H, W] * [B, 1, H, W] -> [B, C, H, W]
        return x * attention_map


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    
    Reference: "CBAM: Convolutional Block Attention Module" (Woo et al., 2018)
    
    CBAM combines SE (channel) attention and spatial attention sequentially.
    This provides both "what" (channel) and "where" (spatial) attention.
    
    Architecture:
    -------------
    Input X
      ↓
    [SE Attention] → X' = SE(X) ⊗ X
      ↓
    [Spatial Attention] → X'' = SA(X') ⊗ X'
      ↓
    Output X''
    
    Why CBAM for VQA:
    ----------------
    VQA benefits from both types of attention:
    - Channel attention: Select relevant feature types (color, texture, shape)
    - Spatial attention: Focus on relevant image regions
    Combined, they enable fine-grained visual understanding.
    """
    
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel: int = 7
    ):
        """
        Initialize CBAM block.
        
        Args:
            channels: Number of input channels
            reduction: SE reduction ratio
            spatial_kernel: Spatial attention kernel size
        """
        super(CBAMBlock, self).__init__()
        
        self.channel_attention = SEAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Channel attention then spatial attention.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Attended tensor [B, C, H, W]
        """
        # Apply channel attention first
        x = self.channel_attention(x)
        
        # Then spatial attention
        x = self.spatial_attention(x)
        
        return x


class SelfAttention2D(nn.Module):
    """
    2D Self-Attention for Feature Maps
    
    This implements self-attention directly on 2D feature maps,
    allowing each spatial location to attend to all other locations.
    
    While more expensive than SE/Spatial attention, it captures
    long-range dependencies that convolutions miss.
    
    Mathematical Formulation:
    -------------------------
    Given input X ∈ R^(C×H×W), reshape to X ∈ R^(C×N) where N=H×W
    
    Q = W_q · X, K = W_k · X, V = W_v · X
    Attention = softmax(Q^T · K / √d)
    Output = V · Attention^T
    
    Note: This is computationally expensive for large feature maps.
    Consider using only in later layers with smaller spatial dimensions.
    """
    
    def __init__(self, channels: int, reduction: int = 8):
        """
        Initialize 2D self-attention.
        
        Args:
            channels: Number of input channels
            reduction: Reduction for query/key dimensions
        """
        super(SelfAttention2D, self).__init__()
        
        self.channels = channels
        self.reduced = channels // reduction
        
        # Query, Key, Value projections
        self.query = nn.Conv2d(channels, self.reduced, 1)
        self.key = nn.Conv2d(channels, self.reduced, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        
        # Learnable scaling factor
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of 2D self-attention.
        
        Args:
            x: Input [B, C, H, W]
            
        Returns:
            Attended output [B, C, H, W]
        """
        batch, channels, height, width = x.shape
        n_pixels = height * width
        
        # Project to Q, K, V
        # [B, C, H, W] -> [B, C', H, W] -> [B, C', N]
        q = self.query(x).view(batch, self.reduced, n_pixels)  # [B, C', N]
        k = self.key(x).view(batch, self.reduced, n_pixels)    # [B, C', N]
        v = self.value(x).view(batch, channels, n_pixels)      # [B, C, N]
        
        # Attention scores: Q^T · K
        # [B, N, C'] × [B, C', N] -> [B, N, N]
        attention = torch.bmm(q.permute(0, 2, 1), k)
        attention = F.softmax(attention / (self.reduced ** 0.5), dim=-1)
        
        # Apply attention to values
        # [B, C, N] × [B, N, N] -> [B, C, N]
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)
        
        # Residual connection with learnable weight
        return self.gamma * out + x


# =============================================================================
# Attention Ablation Wrapper
# =============================================================================

class AttentionWrapper(nn.Module):
    """
    Wrapper for conditional attention application.
    
    Used for ablation studies: easily enable/disable attention
    without modifying the core architecture.
    """
    
    def __init__(
        self,
        channels: int,
        use_se: bool = True,
        use_spatial: bool = True,
        se_reduction: int = 16,
        spatial_kernel: int = 7
    ):
        """
        Initialize attention wrapper.
        
        Args:
            channels: Number of input channels
            use_se: Whether to use SE attention
            use_spatial: Whether to use spatial attention
            se_reduction: SE reduction ratio
            spatial_kernel: Spatial attention kernel size
        """
        super(AttentionWrapper, self).__init__()
        
        self.use_se = use_se
        self.use_spatial = use_spatial
        
        if use_se:
            self.se = SEAttention(channels, se_reduction)
        if use_spatial:
            self.spatial = SpatialAttention(spatial_kernel)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply enabled attention modules."""
        if self.use_se:
            x = self.se(x)
        if self.use_spatial:
            x = self.spatial(x)
        return x


if __name__ == "__main__":
    # Test attention modules
    print("Testing Attention Modules\n" + "=" * 50)
    
    # Test input: batch=2, channels=64, height=14, width=14
    x = torch.randn(2, 64, 14, 14)
    print(f"Input shape: {x.shape}")
    
    # Test SE Attention
    se = SEAttention(channels=64, reduction=16)
    se_out = se(x)
    print(f"SE Attention output shape: {se_out.shape}")
    assert se_out.shape == x.shape, "SE output shape mismatch!"
    
    # Test Spatial Attention
    spatial = SpatialAttention(kernel_size=7)
    spatial_out = spatial(x)
    print(f"Spatial Attention output shape: {spatial_out.shape}")
    assert spatial_out.shape == x.shape, "Spatial output shape mismatch!"
    
    # Test CBAM
    cbam = CBAMBlock(channels=64)
    cbam_out = cbam(x)
    print(f"CBAM output shape: {cbam_out.shape}")
    assert cbam_out.shape == x.shape, "CBAM output shape mismatch!"
    
    # Test 2D Self-Attention
    self_attn = SelfAttention2D(channels=64)
    self_out = self_attn(x)
    print(f"Self-Attention 2D output shape: {self_out.shape}")
    assert self_out.shape == x.shape, "Self-attention output shape mismatch!"
    
    # Test Ablation Wrapper
    wrapper_full = AttentionWrapper(64, use_se=True, use_spatial=True)
    wrapper_se_only = AttentionWrapper(64, use_se=True, use_spatial=False)
    wrapper_none = AttentionWrapper(64, use_se=False, use_spatial=False)
    
    print(f"\nAblation wrapper (full): {wrapper_full(x).shape}")
    print(f"Ablation wrapper (SE only): {wrapper_se_only(x).shape}")
    print(f"Ablation wrapper (none): {wrapper_none(x).shape}")
    
    print("\n✓ All attention module tests passed!")
