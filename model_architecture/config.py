from dataclasses import dataclass

# Global variables will be set in main() function
@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    learning_rate: float
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float
    max_epochs: int
    max_new_tokens: int
    temperature: float
    weight_decay: float = 0.1
    label_smoothing: float = 0.1