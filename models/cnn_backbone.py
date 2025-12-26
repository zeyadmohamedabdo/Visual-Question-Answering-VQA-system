"""
Custom CNN Backbone for VQA System
===================================
This module implements a custom ResNet-like CNN for image feature extraction.

Design Philosophy:
-----------------
We implement the CNN from scratch (no pretrained weights) to satisfy
academic requirements while following proven architectural patterns.

Architecture Overview:
---------------------
1. Stem: Initial convolutions to reduce spatial dimensions
2. Stages 1-4: ResNet-style blocks with increasing channels
3. Integrated attention: SE and Spatial attention in later stages
4. Output: Feature maps (not flattened) for cross-attention

Why ResNet-style:
----------------
- Skip connections enable training deep networks
- Proven effectiveness on vision tasks
- Well-understood gradient flow properties
- Good balance of depth and computational cost

Output Specification:
--------------------
Unlike classification CNNs that output a single vector,
our CNN outputs a feature map [B, 512, 7, 7] that preserves
spatial information. This is crucial for cross-attention,
which needs to attend to different image regions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from models.attention_modules import SEAttention, SpatialAttention, AttentionWrapper


class ConvBlock(nn.Module):
    """
    Basic convolutional block: Conv -> BatchNorm -> ReLU
    
    Why BatchNorm:
    - Stabilizes training by normalizing activations
    - Allows higher learning rates
    - Acts as regularization
    
    Why ReLU:
    - Simple, efficient non-linearity
    - Mitigates vanishing gradient
    - Proven effectiveness
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False
    ):
        """
        Initialize conv block.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding
            bias: Whether to use bias (usually False with BatchNorm)
        """
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input [B, C_in, H, W]
            
        Returns:
            Output [B, C_out, H', W']
        """
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """
    Residual Block with optional downsampling.
    
    Architecture:
    -------------
    x ─┬─> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN ─┬─> + -> ReLU -> out
       │                                           │
       └──────────── (shortcut) ───────────────────┘
    
    If input/output dimensions differ, shortcut uses 1x1 conv.
    
    Why Residual Connections:
    -------------------------
    1. Enable gradient flow through deep networks
    2. Allow learning identity mappings when needed
    3. Prevent degradation in very deep networks
    4. Empirically improves training dynamics
    
    Mathematical Perspective:
    -------------------------
    Instead of learning H(x), we learn F(x) = H(x) - x
    Output: H(x) = F(x) + x
    
    If optimal H(x) ≈ x (identity), then F(x) ≈ 0 is easier to learn
    than learning identity mapping directly.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        """
        Initialize residual block.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            stride: Stride for first conv (>1 for downsampling)
            downsample: Shortcut transformation if dims change
        """
        super(ResidualBlock, self).__init__()
        
        # First conv: may downsample spatially
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second conv: maintains dimensions
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input [B, C_in, H, W]
            
        Returns:
            Output [B, C_out, H', W']
            
        Shape changes when stride > 1:
            H' = H / stride
            W' = W / stride
        """
        # Store identity for skip connection
        identity = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Shortcut (downsample if needed)
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add skip connection
        out += identity
        out = self.relu(out)
        
        return out


class ResidualStage(nn.Module):
    """
    A stage of residual blocks with optional attention.
    
    Each stage consists of:
    1. First block: may downsample (stride=2)
    2. Remaining blocks: maintain dimensions
    3. Optional attention after all blocks
    
    Typical stage configuration:
    - Stage 1: 64 channels, 2 blocks
    - Stage 2: 128 channels, 2 blocks
    - Stage 3: 256 channels, 2 blocks
    - Stage 4: 512 channels, 2 blocks
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 2,
        stride: int = 1,
        use_se: bool = True,
        use_spatial: bool = True,
        se_reduction: int = 16
    ):
        """
        Initialize residual stage.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            num_blocks: Number of residual blocks
            stride: Stride for first block (downsampling)
            use_se: Whether to use SE attention
            use_spatial: Whether to use spatial attention
            se_reduction: SE reduction ratio
        """
        super(ResidualStage, self).__init__()
        
        layers = []
        
        # First block may downsample
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            downsample = None
            
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        self.blocks = nn.Sequential(*layers)
        
        # Attention after residual blocks
        self.attention = AttentionWrapper(
            out_channels,
            use_se=use_se,
            use_spatial=use_spatial,
            se_reduction=se_reduction
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through stage.
        
        Args:
            x: Input [B, C_in, H, W]
            
        Returns:
            Output [B, C_out, H', W']
        """
        x = self.blocks(x)
        x = self.attention(x)
        return x


class CustomResNet(nn.Module):
    """
    Custom ResNet-like CNN Backbone for VQA.
    
    This is a from-scratch implementation following ResNet principles
    but customized for VQA feature extraction.
    
    Architecture:
    -------------
    Input: [B, 3, 224, 224]
    
    Stem:
      Conv 7×7, stride 2 → [B, 64, 112, 112]
      MaxPool 3×3, stride 2 → [B, 64, 56, 56]
    
    Stage 1: 2× ResBlock(64) + SE → [B, 64, 56, 56]
    Stage 2: 2× ResBlock(128), stride 2 + SE → [B, 128, 28, 28]
    Stage 3: 2× ResBlock(256), stride 2 + SE + Spatial → [B, 256, 14, 14]
    Stage 4: 2× ResBlock(512), stride 2 + SE + Spatial → [B, 512, 7, 7]
    
    Output: [B, 512, 7, 7] = 49 spatial locations × 512 features
    
    Key Design Choices:
    -------------------
    1. NO global average pooling - preserve spatial information
    2. Attention in later stages - more semantic features
    3. Output feature map - enables region-level cross-attention
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_blocks: List[int] = [2, 2, 2, 2],
        use_se: bool = True,
        use_spatial: bool = True,
        se_reduction: int = 16
    ):
        """
        Initialize CustomResNet.
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            base_channels: Channels in first stage (doubles each stage)
            num_blocks: Number of residual blocks per stage
            use_se: Whether to use SE attention (for ablation)
            use_spatial: Whether to use spatial attention (for ablation)
            se_reduction: SE reduction ratio
        """
        super(CustomResNet, self).__init__()
        
        self.use_se = use_se
        self.use_spatial = use_spatial
        
        # Channel progression: 64 -> 128 -> 256 -> 512
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8
        ]
        
        # =====================================================================
        # STEM: Initial feature extraction
        # Large 7×7 kernel captures more context in first layer
        # Stride 2 + MaxPool reduces spatial dimensions early
        # =====================================================================
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # After stem: [B, 64, 56, 56]
        
        # =====================================================================
        # STAGE 1: Low-level features (edges, textures)
        # SE attention only - spatial attention less useful here
        # =====================================================================
        self.stage1 = ResidualStage(
            channels[0], channels[0],
            num_blocks=num_blocks[0],
            stride=1,  # No downsampling
            use_se=use_se,
            use_spatial=False,  # Skip spatial at early stage
            se_reduction=se_reduction
        )
        # After stage1: [B, 64, 56, 56]
        
        # =====================================================================
        # STAGE 2: Mid-level features (parts, simple shapes)
        # SE attention, preparing for more semantic features
        # =====================================================================
        self.stage2 = ResidualStage(
            channels[0], channels[1],
            num_blocks=num_blocks[1],
            stride=2,  # Downsample 56 -> 28
            use_se=use_se,
            use_spatial=False,
            se_reduction=se_reduction
        )
        # After stage2: [B, 128, 28, 28]
        
        # =====================================================================
        # STAGE 3: High-level features (objects, semantic)
        # Both SE and Spatial attention
        # =====================================================================
        self.stage3 = ResidualStage(
            channels[1], channels[2],
            num_blocks=num_blocks[2],
            stride=2,  # Downsample 28 -> 14
            use_se=use_se,
            use_spatial=use_spatial,
            se_reduction=se_reduction
        )
        # After stage3: [B, 256, 14, 14]
        
        # =====================================================================
        # STAGE 4: Highest-level features (scene understanding)
        # Both attention types, final feature extraction
        # =====================================================================
        self.stage4 = ResidualStage(
            channels[2], channels[3],
            num_blocks=num_blocks[3],
            stride=2,  # Downsample 14 -> 7
            use_se=use_se,
            use_spatial=use_spatial,
            se_reduction=se_reduction
        )
        # After stage4: [B, 512, 7, 7]
        
        # Store output channels for downstream modules
        self.output_channels = channels[3]
        self.output_spatial_size = 7  # 224 / 32 = 7
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize network weights.
        
        Following best practices:
        - Conv: Kaiming/He initialization (good for ReLU)
        - BatchNorm: Ones for weight, zeros for bias
        - Linear: Xavier/Glorot initialization
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN backbone.
        
        Args:
            x: Input images [B, 3, 224, 224]
            
        Returns:
            Feature maps [B, 512, 7, 7]
            
        Shape tracking:
            Input:   [B, 3, 224, 224]
            Stem:    [B, 64, 56, 56]
            Stage 1: [B, 64, 56, 56]
            Stage 2: [B, 128, 28, 28]
            Stage 3: [B, 256, 14, 14]
            Stage 4: [B, 512, 7, 7]
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x
    
    def get_feature_map_size(self) -> Tuple[int, int, int]:
        """
        Get output feature map dimensions.
        
        Returns:
            Tuple of (channels, height, width)
        """
        return (self.output_channels, self.output_spatial_size, self.output_spatial_size)


def create_cnn_backbone(
    use_attention: bool = True,
    se_reduction: int = 16
) -> CustomResNet:
    """
    Factory function to create CNN backbone.
    
    Args:
        use_attention: If False, disable all attention (for ablation)
        se_reduction: SE reduction ratio
        
    Returns:
        CustomResNet instance
    """
    return CustomResNet(
        use_se=use_attention,
        use_spatial=use_attention,
        se_reduction=se_reduction
    )


if __name__ == "__main__":
    # Test CNN backbone
    print("Testing Custom CNN Backbone\n" + "=" * 50)
    
    # Create model
    model = CustomResNet()
    
    # Test input
    x = torch.randn(2, 3, 224, 224)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected: [2, 512, 7, 7]")
    
    # Verify output shape
    assert output.shape == (2, 512, 7, 7), "Output shape mismatch!"
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test ablation (no attention)
    print("\nTesting ablation (no attention)...")
    model_no_attn = create_cnn_backbone(use_attention=False)
    output_no_attn = model_no_attn(x)
    print(f"No-attention output shape: {output_no_attn.shape}")
    
    no_attn_params = sum(p.numel() for p in model_no_attn.parameters())
    print(f"Parameters without attention: {no_attn_params:,}")
    print(f"Attention adds: {total_params - no_attn_params:,} parameters")
    
    print("\n✓ CNN backbone tests passed!")
