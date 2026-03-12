from dataclasses import dataclass

import torch


@dataclass
class Cache:
    key: torch.Tensor  # (batch_size, seq_len, d_model)
    value: torch.Tensor  # (batch_size, seq_len, d_model)


@dataclass
class Hypothesis:
    token: list[int]
    total_score: float
    caches: list[Cache]
