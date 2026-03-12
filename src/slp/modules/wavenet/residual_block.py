import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class ResidualBlock(nn.Module):
    """Residual block for WaveNet.

    Proposed in A. van den Oord et al., "WaveNet: A Generative Model for Raw Audio," in CoRR, 2016.

    """

    def __init__(self, input_size: int, kernel_size: int, dilation: int, dropout_rate: float, cond_size: int = 0):
        super().__init__()
        assert kernel_size % 2 == 1
        self.input_size = input_size
        self.dilated_conv = weight_norm(
            nn.Conv1d(
                input_size,
                2 * input_size,
                kernel_size,
                dilation=dilation,
                padding=(kernel_size * dilation - dilation) // 2,
            )
        )
        self.res_conv = weight_norm(nn.Conv1d(input_size, input_size, 1))
        self.skip_conv = weight_norm(nn.Conv1d(input_size, input_size, 1))
        self.dropout = nn.Dropout(dropout_rate)

        if cond_size > 0:
            self.cond_conv = weight_norm(nn.Conv1d(cond_size, 2 * input_size, 1))

    def forward(self, x: torch.Tensor, g: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            x (torch.Tensor): Input tensor (batch_size, input_size, seq_length).
            g (torch.Tensor | None): Global conditioning tensor (batch_size, cond_size, seq_length).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                residual (torch.Tensor): Residual output (batch_size, input_size, seq_length).
                skip (torch.Tensor): Skip output (batch_size, input_size, seq_length).
        """
        h = self.dilated_conv(x)

        if g is not None:
            h = h + self.cond_conv(g)

        t, s = torch.split(h, [self.input_size, self.input_size], dim=1)
        h = torch.tanh(t) * torch.sigmoid(s)
        h = self.dropout(h)

        residual = x + self.res_conv(h)
        skip = self.skip_conv(h)

        return residual, skip
