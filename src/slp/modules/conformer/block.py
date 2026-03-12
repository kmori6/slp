import torch
import torch.nn as nn

from slp.modules.conformer.convolution import ConvolutionModule
from slp.modules.conformer.feed_forward import FeedForwardModule
from slp.modules.conformer.multi_head_self_attention import MultiHeadSelfAttentionModule


class Block(nn.Module):
    def __init__(self, input_size: int, num_heads: int, kernel_size: int, dropout_rate: float):
        super().__init__()
        self.ffn1 = FeedForwardModule(input_size, 4 * input_size, dropout_rate)
        self.mhsa = MultiHeadSelfAttentionModule(input_size, num_heads, dropout_rate)
        self.conv = ConvolutionModule(input_size, kernel_size, dropout_rate)
        self.ffn2 = FeedForwardModule(input_size, 4 * input_size, dropout_rate)
        self.layernorm = nn.LayerNorm(input_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Embedding tensor (batch_size, frame_length, input_size).
            mask (torch.Tensor): Mask tensor (batch_size, frame_length, frame_length).

        Returns:
            torch.Tensor: Output tensor (batch_size, frame_length, input_size).
        """
        x = x + 0.5 * self.ffn1(x)
        x = x + self.mhsa(x, mask)
        x = x + self.conv(x)
        x = self.layernorm(x + 0.5 * self.ffn2(x))
        return x
