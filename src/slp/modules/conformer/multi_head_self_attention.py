import torch
import torch.nn as nn

from slp.modules.transformer.relative_positional_multi_head_attention import RelativePositionalMultiHeadAttention


class MultiHeadSelfAttentionModule(RelativePositionalMultiHeadAttention):
    def __init__(self, input_size: int, num_heads: int, dropout_rate: float):
        super().__init__(input_size, input_size // num_heads, num_heads, dropout_rate)
        self.layernorm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """

        Args:
            x (torch.Tensor): Embedding tensor (batch_size, frame_length, input_size).
            mask (torch.Tensor): Mask tensor (batch_size, frame_length, frame_length).

        Returns:
            torch.Tensor: Output tensor (batch_size, frame_length, input_size).
        """
        x = self.layernorm(x)
        x = super().forward(x, x, x, mask)
        x = self.dropout(x)
        return x
