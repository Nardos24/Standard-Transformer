import torch
import argparse
from pathlib import Path
from tokenizers import Tokenizer
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from utils.model_utils import setup_device, load_model, decode_ids, set_seed
from model_architecture.config import GPTConfig
from data_preparation.config import vocab_size


local_rank, device, use_ddp = setup_device()


def generate_text(
    model,
    config,
    max_new_tokens,
    temperature,
    device=None,
):
    model.eval()

    # Start from BOS token
    input_tensor = torch.tensor(
        [[0]],
        device=device
    )

    for _ in range(max_new_tokens):

        with torch.no_grad():
            logits, _ = model(input_tensor)

        # Use only last token prediction
        logits = logits[:, -1, :] / temperature

        # No top-k filtering
        probs = F.softmax(
            logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probs,
            num_samples=1
        )

        input_tensor = torch.cat(
            (input_tensor, next_token),
            dim=1
        )

    return input_tensor[0]


def text_generation(
    model,
    config,
    device=None,
    max_samples=2,
    max_new_tokens=200,
    temperature=0.8,
    tokenizer=None,
):

    decoded_outputs = []

    for sample_idx in range(max_samples):

        generated_ids = generate_text(
            model=model,
            config=config,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
        )

        generated_str = decode_ids(
            tokenizer,
            generated_ids.tolist(),
            stop_at_eos=True
        )

        print(f"\n[Sample {sample_idx + 1}]")
        print(f"[GENERATED]: {generated_str}")

        decoded_outputs.append(generated_str)

    return decoded_outputs



def main():

    set_seed(42)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--flash",
        action="store_true",
        help="Enable FlashAttention"
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=2
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=200
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8
    )

    args = parser.parse_args()


    if use_ddp and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl"
        )


    print(
        f"[Rank {local_rank}] Using device: {device}"
    )


    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=32,
        learning_rate=3e-4,
        n_embd=128,
        n_head=8,
        n_layer=4,
        dropout=0.0,
        max_epochs=5,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )


    model_path = "checkpoints/final_model.pt"

    model = load_model(
        model_path,
        config
    )

    model = model.to(device)


    if use_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank
        )


    tokenizer_path = Path(
        "data_preparation/tokenizer.json"
    )

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}"
        )


    tokenizer = Tokenizer.from_file(
        str(tokenizer_path)
    )


    if not dist.is_initialized() or dist.get_rank() == 0:

        text_generation(
            model=model,
            config=config,
            device=device,
            max_samples=args.max_samples,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            tokenizer=tokenizer,
        )


    if use_ddp and dist.is_initialized():

        dist.barrier()

        dist.destroy_process_group()



if __name__ == "__main__":
    main()