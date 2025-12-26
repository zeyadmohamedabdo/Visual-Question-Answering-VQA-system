
import torch
import torch.nn as nn
import torch.optim as optim
from models.vqa_model import create_vqa_model
import sys
import random
import numpy as np

# Set seed
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

def run_repro():
    print("="*60)
    print("VQA Model Learning Capability Test")
    print("="*60)

    # 1. Create Model
    vocab_size = 100
    num_answers = 10
    embed_dim = 32 # Small dim for speed
    
    print("Creating model...")
    model = create_vqa_model(
        vocab_size=vocab_size,
        num_answers=num_answers,
        embed_dim=embed_dim,
        use_attention=True
    )
    
    # 2. Create Synthetic Batch
    batch_size = 4
    # Random images: [B, 3, 224, 224]
    images = torch.randn(batch_size, 3, 224, 224)
    # Random tokens: [B, 10]
    token_ids = torch.randint(0, vocab_size, (batch_size, 10))
    # Random mask
    attention_mask = torch.ones(batch_size, 10)
    # Fixed target: All answers are class '1'
    targets = torch.tensor([1] * batch_size, dtype=torch.long)
    
    # 3. Training Loop (Overfit)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    print("\nStarting overfitting test (User Reported Acc: 0.008)...")
    model.train()
    
    for step in range(50):
        optimizer.zero_grad()
        
        logits, _ = model(images, token_ids, attention_mask)
        loss = criterion(logits, targets)
        
        loss.backward()
        optimizer.step()
        
        # Check accuracy
        pred = logits.argmax(dim=-1)
        acc = (pred == targets).float().mean().item()
        
        if step % 10 == 0:
            print(f"Step {step:02d}: Loss={loss.item():.4f}, Acc={acc:.4f}")
            
    print(f"Final Step: Loss={loss.item():.4f}, Acc={acc:.4f}")
    
    if acc > 0.9:
        print("\n[SUCCESS] Model is capable of learning!")
        print("The issue is likely in the Dataset, Data Loading, or Hyperparameters.")
    else:
        print("\n[FAILURE] Model failed to overfit a simple batch.")
        print("The issue is likely in the Model Architecture or Gradient Flow.")

if __name__ == "__main__":
    run_repro()
