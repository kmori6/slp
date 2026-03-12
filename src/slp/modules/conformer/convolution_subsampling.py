import torch
import torch.nn as nn


class ConvolutionSubsampling(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, output_size, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(output_size, output_size, kernel_size=3, stride=2)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            x (torch.Tensor): Embedding tensor (batch_size, frame_length, input_size).
            lengths (torch.Tensor): Valid frame lengths (batch_size,).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                torch.Tensor: Output sequence (batch_size, frame_size', output_size').
                torch.Tensor: Mask sequence (batch_size, frame_size').
                where:
                    - frame_size' = (frame_size - 1) // 2 - 1) // 2.
                    - output_size' = output_size * ((input_size - 1) // 2 - 1) // 2.
        """
        x = x[:, None, :, :]  # (batch_size, 1, frame_length, input_size)
        x = self.conv1(x)  # (batch_size, output_size, (frame_length - 1) // 2, (input_size - 1) // 2)
        x = self.activation(x)
        # (batch_size, output_size, ((frame_length - 1) // 2 - 1) // 2, ((input_size - 1) // 2 - 1) // 2)
        x = self.conv2(x)
        x = self.activation(x)
        # (batch_size, ((frame_length - 1) // 2 - 1) // 2, output_size * ((input_size - 1) // 2 - 1) // 2)
        x = x.transpose(1, 2).flatten(2, -1)
        lengths = ((lengths - 1) // 2 - 1) // 2
        lengths = lengths.clamp(min=0)
        return x, torch.arange(x.shape[1], device=lengths.device) < lengths[:, None]
