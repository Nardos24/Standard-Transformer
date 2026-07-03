import os
import math
import time
import torch
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.model_utils import cleanup_memory, setup_device, set_seed
from model_architecture.config import GPTConfig
from model_architecture.model import LanguageModel
from data_preparation.dataloader import get_loaders
from data_preparation.config import vocab_size
from eval import evaluate
from visualization import plot_metrics

# Training Step
def train(model, train_loader, optimizer, scheduler, device):
    model.train()
    total_loss, total_batches = 0, 0

    # Sync GPU for accurate timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for batch_idx, batch in enumerate(train_loader):
        xb = batch['input_ids'].to(device)
        yb = batch['target_ids'].to(device)

        _, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_batches += 1
        perplexity = torch.exp(loss).item()

        # Print status from rank 0 every 10 batches
        if (not dist.is_initialized() or dist.get_rank() == 0) and (batch_idx + 1) % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"  Batch {batch_idx + 1}/{len(train_loader)} | "
                f"Train Loss {loss:.4f} | Train Perplexity {perplexity:.4f} | "
                f"LR {current_lr:.6f}"
            )

        cleanup_memory()

    # Sync before returning stats
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    avg_loss = total_loss / total_batches
    avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return avg_loss, avg_perplexity

# Main Training Loop
def get_param_groups(model, weight_decay):
    """
    Separate parameters into two groups:
    - 2D weight tensors (linear layers, embeddings) get weight decay
    - Everything else (biases, LayerNorm weights/biases) gets no weight decay
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Apply weight decay only to 2D+ weight matrices (not biases or norm params)
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    Cosine annealing LR schedule with linear warmup.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--flash', action='store_true', 
                        help='Enable FlashAttention (not implemented in this model)')
    
    parser.parse_args()  
    
    # Setup device and DDP
    local_rank, device, use_ddp = setup_device()

    if use_ddp and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        
    # Model configuration
    config = GPTConfig(
        vocab_size = vocab_size,
        block_size = 32, 
        learning_rate = 3e-4,
        n_embd=128,
        n_head = 8,
        n_layer = 4,
        dropout= 0.2,
        max_epochs = 5,
        max_new_tokens = 200,
        temperature = 0.8,
        weight_decay = 0.1,
        label_smoothing = 0.1
    )

    # Model
    model = LanguageModel(config).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Data
    train_loader, valid_loader, _ = get_loaders(distributed=use_ddp)

    # Optimizer with proper parameter group separation
    param_groups = get_param_groups(model, config.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=config.learning_rate, betas=(0.9, 0.95))

    # Cosine LR schedule with linear warmup (10% of total steps)
    total_steps = len(train_loader) * config.max_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    train_losses = []
    val_losses = []
    train_perplexities = []
    val_perplexities = []  

    rank = dist.get_rank() if dist.is_initialized() else 0
    best_val_loss = float('inf')
    
    # Print model info from rank 0
    if rank == 0:
        print("========== Starting training ==========")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"{num_params:.5f}M parameters")
        print(f"Dropout: {config.dropout} | Weight Decay: {config.weight_decay} | "
              f"Label Smoothing: {config.label_smoothing}")
        print(f"Total steps: {total_steps} | Warmup steps: {warmup_steps}")
      
    start_time = time.time()
    
    # Training epochs 
    for epoch in range(config.max_epochs):

        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\nEpoch {epoch + 1}/{config.max_epochs}")
        
        avg_loss, avg_perplexity = train(model, train_loader, optimizer, scheduler, device)
        train_losses.append(avg_loss)
        train_perplexities.append(avg_perplexity)
        
        with torch.no_grad():
                val_loss, val_perplexity = evaluate(
                    model, valid_loader, max_batches=None, device=device
                ) 
        val_losses.append(val_loss)
        val_perplexities.append(val_perplexity)
            
        if rank == 0:
            print(
                f"Epoch {epoch + 1}/{config.max_epochs} | "
                f"Train Loss: {avg_loss:.4f} | Train Perplexity: {avg_perplexity:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Perplexity: {val_perplexity:.4f}"
            )

            # Save best-val checkpoint
            os.makedirs("checkpoints", exist_ok=True)
            model_state = model.module.state_dict() if use_ddp else model.state_dict()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({"model_state": model_state}, "checkpoints/best_model.pt")
                print(f"  ✓ New best val loss: {val_loss:.4f} — checkpoint saved.")

            # Also save latest checkpoint (for resuming)
            torch.save({"model_state": model_state}, "checkpoints/final_model.pt")
            
    # Final summary
    if rank == 0:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time = time.time() - start_time
        print(f"\nTotal Training Time: {total_time:.2f} seconds")
        print(f"Best Val Loss: {best_val_loss:.4f}")
        print("========== Training completed ==========")

        plot_metrics(
            train_losses,
            val_losses,
            train_perplexities,
            val_perplexities
        )
        
        print("Model saved.")

    # Cleanup
    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()