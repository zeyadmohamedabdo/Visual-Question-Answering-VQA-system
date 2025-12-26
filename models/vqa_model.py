"""
Complete VQA Model
==================
This module assembles all components into the final VQA model.

Architecture Overview:
---------------------
1. Image Encoder (CustomResNet): Extract visual features
2. Text Encoder (TransformerEncoder): Encode question
3. Multimodal Fusion: Cross-attention based fusion
4. Answer Head: Classification over top-1000 answers

The model takes an image and question as input,
and outputs logits over the answer vocabulary.

This is the main entry point for training and inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

# Import all model components
from models.cnn_backbone import CustomResNet, create_cnn_backbone
from models.text_encoder import TransformerTextEncoder
from models.fusion import MultimodalFusion


class AnswerHead(nn.Module):
    """
    Answer Classification Head
    
    Takes fused multimodal features and predicts answer class.
    
    Architecture:
    -------------
    Fused Features [B, D]
         ↓
    FC -> ReLU -> Dropout
         ↓
    FC -> ReLU -> Dropout
         ↓
    FC -> Logits [B, num_answers]
         ↓
    Softmax (during inference)
    
    Why MLP (not single linear):
    ---------------------------
    1. Additional capacity for answer prediction
    2. Non-linear transformation of fused features
    3. Dropout for regularization
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_answers: int,
        dropout: float = 0.3
    ):
        """
        Initialize answer head.
        
        Args:
            input_dim: Input feature dimension (from fusion)
            hidden_dim: Hidden layer dimension
            num_answers: Number of answer classes (1000)
            dropout: Dropout rate (higher for classification head)
        """
        super(AnswerHead, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, num_answers)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize classifier weights."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict answer logits.
        
        Args:
            x: Fused features [B, input_dim]
            
        Returns:
            Answer logits [B, num_answers]
        """
        return self.classifier(x)


class VQAModel(nn.Module):
    """
    Complete Visual Question Answering Model
    
    This model combines:
    1. CustomResNet for image encoding
    2. TransformerEncoder for question encoding
    3. Cross-attention fusion for multimodal interaction
    4. MLP classifier for answer prediction
    
    The model is trained end-to-end from scratch,
    satisfying the academic requirement of no pretrained components.
    
    Input:
    ------
    - images: [B, 3, 224, 224] - normalized RGB images
    - token_ids: [B, seq_len] - tokenized question indices
    - attention_mask: [B, seq_len] - padding mask (optional)
    
    Output:
    -------
    - logits: [B, num_answers] - answer class logits
    - aux_outputs: dict - attention maps and intermediate features
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 256,
        num_answers: int = 1000,
        # Image encoder config
        use_se_attention: bool = True,
        use_spatial_attention: bool = True,
        se_reduction: int = 16,
        # Text encoder config
        num_transformer_layers: int = 4,
        num_attention_heads: int = 8,
        ffn_hidden_dim: int = 1024,
        max_question_length: int = 20,
        # Fusion config
        num_cross_layers: int = 2,
        use_gating: bool = True,
        # General
        dropout: float = 0.1,
        answer_dropout: float = 0.3
    ):
        """
        Initialize VQA model.
        
        Args:
            vocab_size: Question vocabulary size
            embed_dim: Embedding dimension for text/fusion
            num_answers: Number of answer classes
            use_se_attention: Use SE attention in CNN (ablation)
            use_spatial_attention: Use spatial attention in CNN (ablation)
            se_reduction: SE reduction ratio
            num_transformer_layers: Number of transformer layers
            num_attention_heads: Number of attention heads
            ffn_hidden_dim: Transformer FFN hidden dim
            max_question_length: Maximum question tokens
            num_cross_layers: Cross-attention layers in fusion
            use_gating: Use gating in fusion
            dropout: General dropout rate
            answer_dropout: Dropout in answer head
        """
        super(VQAModel, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_answers = num_answers
        
        # =====================================================================
        # Image Encoder: Custom ResNet with attention
        # =====================================================================
        self.image_encoder = CustomResNet(
            use_se=use_se_attention,
            use_spatial=use_spatial_attention,
            se_reduction=se_reduction
        )
        cnn_channels = self.image_encoder.output_channels  # 512
        cnn_spatial = self.image_encoder.output_spatial_size  # 7
        
        # =====================================================================
        # Text Encoder: Transformer
        # =====================================================================
        self.text_encoder = TransformerTextEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_transformer_layers,
            num_heads=num_attention_heads,
            ffn_hidden_dim=ffn_hidden_dim,
            max_length=max_question_length,
            dropout=dropout,
            pad_idx=0  # Assuming 0 is PAD token
        )
        
        # =====================================================================
        # Multimodal Fusion: Cross-attention + gating
        # =====================================================================
        self.fusion = MultimodalFusion(
            image_channels=cnn_channels,
            image_spatial_size=cnn_spatial,
            embed_dim=embed_dim,
            num_heads=num_attention_heads,
            num_cross_layers=num_cross_layers,
            dropout=dropout,
            use_gating=use_gating
        )
        
        # =====================================================================
        # Answer Head: Classification
        # =====================================================================
        self.answer_head = AnswerHead(
            input_dim=embed_dim,
            hidden_dim=embed_dim * 2,  # 512
            num_answers=num_answers,
            dropout=answer_dropout
        )
        
        # Store config for saving/loading
        self.config = {
            'vocab_size': vocab_size,
            'embed_dim': embed_dim,
            'num_answers': num_answers,
            'use_se_attention': use_se_attention,
            'use_spatial_attention': use_spatial_attention,
            'se_reduction': se_reduction,
            'num_transformer_layers': num_transformer_layers,
            'num_attention_heads': num_attention_heads,
            'ffn_hidden_dim': ffn_hidden_dim,
            'max_question_length': max_question_length,
            'num_cross_layers': num_cross_layers,
            'use_gating': use_gating,
            'dropout': dropout,
            'answer_dropout': answer_dropout
        }
        
    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass of VQA model.
        
        Args:
            images: Input images [B, 3, 224, 224]
            token_ids: Question token indices [B, seq_len]
            attention_mask: Question padding mask [B, seq_len]
            return_aux: Whether to return auxiliary outputs
            
        Returns:
            logits: Answer logits [B, num_answers]
            aux_outputs: Intermediate features (if return_aux=True)
            
        Shape tracking through the network:
            images:          [B, 3, 224, 224]
            image_features:  [B, 512, 7, 7]
            token_ids:       [B, seq_len]
            text_features:   [B, seq_len, 256]
            fused:           [B, 256]
            logits:          [B, 1000]
        """
        # =====================================================================
        # Step 1: Encode image
        # CustomResNet extracts visual features preserving spatial structure
        # =====================================================================
        # [B, 3, 224, 224] -> [B, 512, 7, 7]
        image_features = self.image_encoder(images)
        
        # =====================================================================
        # Step 2: Encode question
        # Transformer produces contextualized token representations
        # =====================================================================
        # [B, seq_len] -> [B, seq_len, 256], [B, 256]
        text_features, text_pooled = self.text_encoder(token_ids, attention_mask)
        
        # =====================================================================
        # Step 3: Fuse modalities
        # Cross-attention allows question to attend to relevant image regions
        # =====================================================================
        # [B, 512, 7, 7] + [B, seq_len, 256] -> [B, 256]
        fused, fusion_aux = self.fusion(
            image_features, text_features, attention_mask
        )
        
        # =====================================================================
        # Step 4: Predict answer
        # MLP classifier over fused representation
        # =====================================================================
        # [B, 256] -> [B, num_answers]
        logits = self.answer_head(fused)
        
        if return_aux:
            aux_outputs = {
                'image_features': image_features,
                'text_features': text_features,
                'text_pooled': text_pooled,
                'fused': fused,
                **fusion_aux
            }
            return logits, aux_outputs
        
        return logits, None
    
    def predict(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        top_k: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict top-k answers with probabilities.
        
        Args:
            images: Input images [B, 3, 224, 224]
            token_ids: Question tokens [B, seq_len]
            attention_mask: Padding mask [B, seq_len]
            top_k: Number of top predictions to return
            
        Returns:
            top_indices: Top-k answer indices [B, k]
            top_probs: Top-k probabilities [B, k]
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(images, token_ids, attention_mask)
            probs = F.softmax(logits, dim=-1)
            top_probs, top_indices = probs.topk(top_k, dim=-1)
        
        return top_indices, top_probs
    
    def get_attention_maps(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get attention maps for visualization.
        
        Args:
            images: Input images [B, 3, 224, 224]
            token_ids: Question tokens [B, seq_len]
            attention_mask: Padding mask [B, seq_len]
            
        Returns:
            Dictionary with various attention maps
        """
        _, aux = self.forward(images, token_ids, attention_mask, return_aux=True)
        
        # Get cross-attention visualization
        cross_attn_vis = self.fusion.get_attention_visualization(
            aux['cross_attention_weights'],
            spatial_size=self.image_encoder.output_spatial_size
        )
        
        return {
            'cross_attention': aux['cross_attention_weights'],
            'cross_attention_spatial': cross_attn_vis
        }
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get parameter counts for each component."""
        counts = {
            'image_encoder': sum(p.numel() for p in self.image_encoder.parameters()),
            'text_encoder': sum(p.numel() for p in self.text_encoder.parameters()),
            'fusion': sum(p.numel() for p in self.fusion.parameters()),
            'answer_head': sum(p.numel() for p in self.answer_head.parameters()),
        }
        counts['total'] = sum(counts.values())
        return counts


def create_vqa_model(
    vocab_size: int = 10000,
    num_answers: int = 1000,
    use_attention: bool = True,
    **kwargs
) -> VQAModel:
    """
    Factory function to create VQA model.
    
    Args:
        vocab_size: Question vocabulary size
        num_answers: Number of answer classes
        use_attention: If False, disable CNN attention (ablation)
        **kwargs: Additional config overrides
        
    Returns:
        Configured VQAModel instance
    """
    return VQAModel(
        vocab_size=vocab_size,
        num_answers=num_answers,
        use_se_attention=use_attention,
        use_spatial_attention=use_attention,
        **kwargs
    )


def load_vqa_model(checkpoint_path: str, device: str = 'cpu') -> VQAModel:
    """
    Load VQA model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to
        
    Returns:
        Loaded VQAModel
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint
    config = checkpoint.get('config', {})
    
    # Create model with saved config
    model = VQAModel(**config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model.to(device)


if __name__ == "__main__":
    # Test complete VQA model
    print("Testing Complete VQA Model\n" + "=" * 60)
    
    # Create model
    model = VQAModel(
        vocab_size=10000,
        embed_dim=256,
        num_answers=1000
    )
    
    # Test input
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    token_ids = torch.randint(0, 10000, (batch_size, 20))
    attention_mask = torch.ones(batch_size, 20)
    attention_mask[0, 15:] = 0  # Simulate padding
    
    print(f"Input shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Token IDs: {token_ids.shape}")
    print(f"  Attention mask: {attention_mask.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits, aux = model(images, token_ids, attention_mask, return_aux=True)
    
    print(f"\nOutput shapes:")
    print(f"  Logits: {logits.shape}")
    print(f"  Expected: [{batch_size}, 1000]")
    
    # Check auxiliary outputs
    print(f"\nAuxiliary outputs:")
    print(f"  Image features: {aux['image_features'].shape}")
    print(f"  Text features: {aux['text_features'].shape}")
    print(f"  Fused: {aux['fused'].shape}")
    
    # Test prediction
    top_indices, top_probs = model.predict(images, token_ids, attention_mask, top_k=5)
    print(f"\nTop-5 predictions:")
    print(f"  Indices shape: {top_indices.shape}")
    print(f"  Probabilities shape: {top_probs.shape}")
    print(f"  Sample probs: {top_probs[0].tolist()}")
    
    # Parameter counts
    param_counts = model.get_num_parameters()
    print(f"\nParameter counts:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")
    
    # Verify output shape
    assert logits.shape == (batch_size, 1000), "Output shape mismatch!"
    
    # Test ablation (no attention)
    print("\nTesting ablation (no CNN attention)...")
    model_no_attn = create_vqa_model(
        vocab_size=10000,
        num_answers=1000,
        use_attention=False
    )
    logits_no_attn, _ = model_no_attn(images, token_ids, attention_mask)
    print(f"  No-attention output shape: {logits_no_attn.shape}")
    
    no_attn_params = model_no_attn.get_num_parameters()
    print(f"  Parameters: {no_attn_params['total']:,}")
    print(f"  Attention adds: {param_counts['total'] - no_attn_params['total']:,} params")
    
    print("\n✓ Complete VQA model tests passed!")
