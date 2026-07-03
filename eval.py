import os
import sys
import time
import torch
import argparse
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from data_preparation.dataloader import get_loaders
from utils.model_utils import setup_device, load_model, set_seed
from model_architecture.config import GPTConfig
from data_preparation.config import vocab_size

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

local_rank, device, use_ddp = setup_device()

@torch.no_grad()
def evaluate(model, test_loader, max_batches=None, device=None):
    start_time = time.time()
    model.eval()
    
    total_loss = 0
    total_batches = 0

    if not dist.is_initialized() or dist.get_rank() == 0:
        if max_batches is None:
            print(f"Evaluating on the full test set...")
        else:
            print(f"Evaluating on up to {max_batches} batches...")

    # Synchronize before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for batch_idx, batch in enumerate(test_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        input_ids = batch['input_ids'].to(device)
        targets = batch['target_ids'].to(device)

        _, loss = model(input_ids, targets)
        
        total_loss += loss.item()
        total_batches += 1
        
        perplexity = torch.exp(loss).item()

        if (not dist.is_initialized() or dist.get_rank() == 0) and (batch_idx + 1) % 10 == 0:
            print(
                f"  Batch {batch_idx + 1}/{len(test_loader)} | "
                f"Test Loss {loss:.4f} | Test Perplexity {perplexity:.4f}",
                flush=True
            )

    # End timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    avg_loss = total_loss / total_batches if total_batches > 0 else float('inf')
    avg_perplexity = torch.exp(torch.tensor(avg_loss)).item() if avg_loss != float('inf') else float('inf')
    
    elapsed = time.time() - start_time
    
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Evaluation completed in {elapsed:.2f} seconds")
        print(f"Total Batches Processed: {batch_idx + 1}")
        print(f"Avg Test CE Loss: {avg_loss:.4f} | Avg Test Perplexity: {avg_perplexity:.4f}")
    
    return avg_loss,avg_perplexity
    
def main():
    set_seed(42)
    """Entry point for evaluating the predictive coding transformer model."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--flash', action='store_true', help='Enable FlashAttention for attention layers')
    parser.parse_args() 

    # Initialize DDP
    if use_ddp and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
            
    # Model config
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

    # Load model
    model_path = "checkpoints/final_model.pt"
    model = load_model(model_path,config).to(device)
    
    # Wrap in DDP
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Load data
    _, _, test_loader = get_loaders(distributed=use_ddp)

    # Sync before timing
    if torch.cuda.is_available():
            torch.cuda.synchronize()
        
    # Run evaluation
    avg_loss, avg_perplexity = evaluate(
        model, 
        test_loader, 
        max_batches = None, 
        device = device
    )
        
    # Sync after evaluation
    if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Cleanup DDP
    if use_ddp and dist.is_initialized():   
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main() 
