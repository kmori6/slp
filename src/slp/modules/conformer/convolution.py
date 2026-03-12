import torch
import torch.nn as nn


class ConvolutionModule(nn.Module):
    def __init__(self, input_size: int, kernel_size: int, dropout_rate: float):
        super().__init__()
        self.layernorm = nn.LayerNorm(input_size)
        self.pointwise_conv1 = nn.Conv1d(input_size, 2 * input_size, kernel_size=1, stride=1, padding=0)
        self.glu_activation = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            input_size,
            input_size,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=input_size,
        )
        self.batchnorm = nn.BatchNorm1d(input_size)
        self.swish_activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(input_size, input_size, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Embedding tensor (batch_size, frame_length, input_size).

        Returns:
            torch.Tensor: Output tensor (batch_size, frame_length, input_size).
        """
        x = self.layernorm(x)
        x = x.transpose(1, 2)  # (batch_size, input_size, frame_length)
        x = self.pointwise_conv1(x)  # (batch_size, 2 * input_size, frame_length)
        x = self.glu_activation(x)  # (batch_size, input_size, frame_length)
        x = self.depthwise_conv(x)
        x = self.batchnorm(x)
        x = self.swish_activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)  # (batch_size, frame_length, input_size)
        x = self.dropout(x)
        return x
