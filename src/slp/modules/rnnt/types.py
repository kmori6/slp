from dataclasses import dataclass

import torch


@dataclass
class Sequence:
    token: list[int]
    hidden_state: torch.Tensor
    cell_state: torch.Tensor
    total_score: float
