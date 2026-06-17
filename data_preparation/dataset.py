import torch
from pathlib import Path
from torch.utils.data import Dataset


class EncodedDataset(Dataset):
    """
    Dataset that builds overlapping sliding-window sequences
    for next-token prediction.
    """

    def __init__(self, file_path, block_size, stride=2, pad_token_id=3):
        self.block_size = block_size
        self.stride = block_size

        if not Path(file_path).exists():
            raise FileNotFoundError(f"Tokenized file not found: {file_path}")

        tokens = torch.load(file_path, weights_only=False)

        window_size = block_size + 1

        if len(tokens) < window_size:
            self.sequences = torch.empty(
                (0, window_size),
                dtype=tokens.dtype
            )

        else:
            # Create sliding windows
            self.sequences = tokens.unfold(
                dimension=0,
                size=window_size,
                step=self.stride
            )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        input_ids = seq[:-1].clone().detach()
        target_ids = seq[1:].clone().detach()

        return {
            "input_ids": input_ids,
            "target_ids": target_ids
        }