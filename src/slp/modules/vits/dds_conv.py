import torch
import torch.nn as nn


class DDSConv(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int, dilation: int, dropout_rate: float):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size,
            groups=hidden_size,
            dilation=dilation,
            padding=(kernel_size * dilation - dilation) // 2,
        )
        self.first_norm = nn.LayerNorm(hidden_size)
        self.pointwise_conv = nn.Conv1d(hidden_size, hidden_size, 1)
        self.last_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Input tensor (batch_size, hidden_size, seq_length).

        Returns:
            torch.Tensor: Output tensor (batch_size, hidden_size, seq_length).
        """
        residual = x
        x = self.depthwise_conv(x)
        x = self.first_norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.gelu(x)
        x = self.pointwise_conv(x)
        x = self.last_norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.gelu(x)
        x = self.dropout(x)
        x = residual + x
        return x
