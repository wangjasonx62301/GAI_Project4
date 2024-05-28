from dataclasses import dataclass
import torch

@dataclass
class config:
    batch_size : int = 8
    C : int = 1
    H : int = 128
    W : int = 128
    epochs : int = 500
    generate_interval : int = 50
    n_steps : int = 1000
    time_n_embd : int = 100
    min_beta : float = 0.0001
    max_beta : float = 0.02
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_path = 'data/Tata Safari'
    lr : float = 0.002