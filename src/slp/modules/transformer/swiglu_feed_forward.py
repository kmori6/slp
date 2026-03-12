import torch
import torch.nn as nn

from slp.modules.transformer.feed_forward import FeedForward


class SwiGLUFeedForward(FeedForward):
    """Feed-forward network with Swish-Gated Linear Unit (SwiGLU).

    Propsed in N. Shazeer et al., "GLU variants improve transformer," arXiv preprint arXiv:2002.05202, 2020.

    """

    def __init__(self, input_size: int, hidden_size: int, dropout_rate: float, bias: bool = True):
        super().__init__(input_size, hidden_size, dropout_rate, activation=nn.SiLU(), bias=bias)
        self.w_v = nn.Linear(input_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Embedding tensor (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: Output tensor (batch_size, sequence_length, input_size).
        """
        x = self.activation(self.w_1(x)) * self.w_v(x)  # (batch_size, sequence_length, hidden_size)
        x = self.dropout(x)
        x = self.w_2(x)  # (batch_size, sequence_length, input_size)
        return x
