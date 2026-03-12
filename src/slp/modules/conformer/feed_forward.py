import torch
import torch.nn as nn

from slp.modules.transformer.feed_forward import FeedForward


class FeedForwardModule(FeedForward):
    def __init__(self, input_size: int, hidden_size: int, dropout_rate: float):
        super().__init__(input_size, hidden_size, dropout_rate, activation=nn.SiLU())
        self.layernorm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Embedding tensor (batch_size, frame_length, input_size).

        Returns:
            torch.Tensor: Output tensor (batch_size, frame_length, input_size).
        """
        x = self.layernorm(x)
        x = super().forward(x)
        x = self.dropout(x)
        return x
