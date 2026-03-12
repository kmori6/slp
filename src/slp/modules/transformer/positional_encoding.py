import torch
import torch.nn as nn


def sinusoidal_positional_encoding(d_model: int, max_length: int, base: float = 10000.0) -> torch.Tensor:
    """Sinusoidal positional encoding.

    PE_{(pos, 2i)} = sin(pos/10000^{2i/d_model})
    PE_{(pos, 2i + 1)} = cos(pos/10000^{2i/d_model})

    Args:
        d_model (int): Hidden state dimension.
        max_length (int): Maximum sequence length.
        base (float): Base value for frequency calculation.

    Returns:
        torch.Tensor: Sinusoidal positional encoding (max_length, d_model).
    """
    pos = torch.arange(max_length, dtype=torch.float32)[:, None]
    theta = pos / (base ** (torch.arange(0, d_model, 2, dtype=torch.float32) / d_model))
    pe = torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1).flatten(1, -1)
    return pe


class PositionalEncoding(nn.Module):
    """Positional encoding module.

    Proposed in A. Vaswani et al., "Attention is all you need," in NeurIPS, 2017, pp. 5998-6008.

    """

    def __init__(self, hidden_size: int, max_length: int = 4096, base: float = 10000.0):
        super().__init__()
        assert hidden_size % 2 == 0
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.base = base
        self.register_buffer("pe", sinusoidal_positional_encoding(hidden_size, max_length, base), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Embedding tensor (*, sequence_length, hidden_size).

        Returns:
            torch.Tensor: Positional encoding (1, sequence_length, hidden_size).
        """
        seq_len = x.shape[-2]
        if seq_len > self.max_length:
            self.pe = sinusoidal_positional_encoding(self.hidden_size, seq_len, self.base).to(device=x.device)
            self.max_length = seq_len
        pe = self.pe[None, :seq_len, :]
        if pe.dtype != x.dtype:
            pe = pe.to(dtype=x.dtype)
        return pe
