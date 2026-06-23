import torch
import argparse
from pathlib import Path
from tokenizers import Tokenizer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.model_utils import setup_device, load_model, decode_ids, set_seed
from model_architecture.config import GPTConfig
from data_preparation.config import vocab_size

# Device + DDP setup
local_rank, device, use_ddp = setup_device()

@torch.no_grad()
def generate(model, input_ids, max_new_tokens, temperature, device, top_k=None):
    """Generate text samples from the model"""
    model.eval()
    input_tensor = input_ids.to(device)  

    generated = model.generate(
        input_tensor, 
        max_new_tokens=max_new_tokens, 
        temperature=temperature,
        top_k=top_k 
    )
    
    return generated[0]

def text_generation(model, config, input_ids, device, num_samples, max_new_tokens, temperature, tokenizer=None):
    model.eval()
    decoded_outputs = []
    
    if input_ids is None:
        start_token = getattr(config, "start_token_id", 0)  
        input_tensor = torch.tensor([start_token], device=device).unsqueeze(0)
    else:
        input_tensor = input_ids.unsqueeze(0).to(device)
        
    for i in range(num_samples):
    
        generated_ids = generate(
            model=model, 
            input_ids=input_tensor, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device, 
            top_k=None
        )
        
        # Decode to string
        if tokenizer is not None:
            generated_str = tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True)
        else:
            generated_str = decode_ids(
                generated_ids.tolist(),
                stop_at_eos=True
            )
        
        # Clean formatting
        generated_str = generated_str.replace('<PAD>', '').replace('<EOS>', '').strip()
        decoded_outputs.append(generated_str)
    
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"\n[Sample {i + 1}]")
            print(f"[GENERATED]: {generated_str}")
    
    return decoded_outputs

def main():
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--flash', action='store_true', 
                        help='Enable FlashAttention for attention layers')
    parser.add_argument('--num_samples', type=int, default=2, 
                        help='Number of independent text samples to generate.')
    parser.add_argument('--max_tokens', type=int, default=200, 
                        help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, 
                        help='Sampling temperature')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Optional text prompt to condition generation')
    args = parser.parse_args()

    # Setup DDP if needed
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
        dropout= 0.0,
        max_epochs = 5,
        max_new_tokens = 200,
        temperature = 0.8
    )
    
    # Load model
    model_path = "checkpoints/final_model.pt"
    model = load_model(model_path, config).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    tokenizer_path = Path("data_preparation/tokenizer.json")        
        
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. "
            "Please run 'python data_preparation/prepare_tokens.py' first."
        )
        
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    input_ids = None
    
    if args.prompt:
        encoded = tokenizer.encode(args.prompt)
        input_ids = torch.tensor(encoded.ids, dtype=torch.long)
            
    # Run generation only on rank 0
    if not dist.is_initialized() or dist.get_rank() == 0:
        decoded_outputs = text_generation(
            model, config, input_ids, device,
            num_samples=args.num_samples,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            tokenizer=tokenizer
        )

    # Cleanup
    if use_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()