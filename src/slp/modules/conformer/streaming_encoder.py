import random

import torch

from slp.modules.conformer.encoder import Encoder
from slp.utils.mask import streaming_mask


class StreamingEncoder(Encoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int,
        kernel_size: int,
        num_blocks: int,
        dropout_rate: float,
        min_chunk_size: int,
        max_chunk_size: int,
        streaming_mask_ratio: float = 0.6,
    ):
        super().__init__(input_size, hidden_size, num_heads, kernel_size, num_blocks, dropout_rate)
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.streaming_mask_ratio = streaming_mask_ratio

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            x (torch.Tensor): Acoustic embedding tensor (batch_size, frame_length, input_size).
            lengths (torch.Tensor): Valid frame lengths (batch_size,).

        Returns:
            torch.Tensor: Output embedding tensor (batch_size, frame_length', hidden_size).
            torch.Tensor: Mask tensor (batch_size, frame_length').
            where frame_length' = ((frame_length - 1) // 2 - 1) // 2.
        """
        x, mask = self.convolution_subsampling(
            x, lengths
        )  # (batch_size, frame_length', hidden_size * (((input_size - 1) // 2 - 1) // 2)
        x = self.linear(x)  # (batch_size, frame_length', hidden_size)
        x = self.dropout(x)

        # apply streaming mask only during training with streaming_mask_ratio probability
        if self.training and random.random() < self.streaming_mask_ratio:
            frame_length = x.shape[1]
            chunk_size = random.randint(self.min_chunk_size, self.max_chunk_size)
            history_size = random.randint(0, frame_length)
            block_mask = (
                streaming_mask(x.new_full((1,), frame_length, dtype=torch.long), chunk_size, history_size)[None, :, :]
                & mask[:, None, :]
            )
        else:
            block_mask = mask[:, None, :]  # (batch_size, 1, frame_length') -> broadcast

        for block in self.blocks:
            x = block(x, block_mask)
        return x, mask
