import torch
import torch.nn as nn

from slp.modules.conformer.block import Block
from slp.modules.conformer.convolution_subsampling import ConvolutionSubsampling


class Encoder(nn.Module):
    """Conformer modules.

    Proposed in A. Gulati et al., "Conformer: Convolution-augmented transformer for speech recognition,"
    in Interspeech, 2020, pp. 5036-5040.

    """

    def __init__(
        self, input_size: int, hidden_size: int, num_heads: int, kernel_size: int, num_blocks: int, dropout_rate: float
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.convolution_subsampling = ConvolutionSubsampling(hidden_size)
        self.linear = nn.Linear(hidden_size * (((input_size - 1) // 2 - 1) // 2), hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.blocks = nn.ModuleList(
            [Block(hidden_size, num_heads, kernel_size, dropout_rate) for _ in range(num_blocks)]
        )
        self.conformer_blocks = self.blocks

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            x (torch.Tensor): Embedding tensor (batch_size, frame_length, input_size).
            lengths (torch.Tensor): Valid frame lengths (batch_size,).

        Returns:
            torch.Tensor: Output tensor (batch_size, frame_length', hidden_size).
            torch.Tensor: Mask tensor (batch_size, frame_length').
            where frame_length' = ((frame_length - 1) // 2 - 1) // 2.
        """
        x, mask = self.convolution_subsampling(
            x, lengths
        )  # (batch_size, frame_length', hidden_size * (((input_size - 1) // 2 - 1) // 2)
        x = self.linear(x)  # (batch_size, frame_length', hidden_size)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, mask[:, None, :])
        return x, mask
