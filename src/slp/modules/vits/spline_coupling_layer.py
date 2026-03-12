import math

import torch
import torch.nn as nn

from slp.modules.transforms.rational_quadratic_transforms import rational_quadratic_transforms
from slp.modules.vits.dds_conv import DDSConv


class SplineCouplingLayer(nn.Module):
    """Neural spline coupling layer used inside the stochastic duration predictor."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
        num_blocks: int,
        num_bins: int,
        B: float,
        dropout_rate: float,
    ):
        super().__init__()
        assert input_size % 2 == 0
        self.half_input_size = input_size // 2
        self.hidden_size = hidden_size
        self.num_bins = num_bins
        self.B = B

        self.input_conv = nn.Conv1d(self.half_input_size, hidden_size, 1)
        self.blocks = nn.ModuleList(
            [
                DDSConv(
                    hidden_size=hidden_size,
                    kernel_size=kernel_size,
                    dilation=kernel_size**i,
                    dropout_rate=dropout_rate,
                )
                for i in range(num_blocks)
            ]
        )
        self.output_conv = nn.Conv1d(hidden_size, self.half_input_size * (num_bins * 3 - 1), 1)

        # NOTE: Zero-initialized so the layer starts as an identity transform (Glow; Kingma & Dhariwal, 2018).
        nn.init.zeros_(self.output_conv.weight)
        if self.output_conv.bias is not None:
            nn.init.zeros_(self.output_conv.bias)

    def forward(
        self, x: torch.Tensor, g: torch.Tensor | None = None, reverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            x (torch.Tensor): Input tensor (batch_size, input_size, seq_length).
            g (torch.Tensor | None): Optional conditioning tensor (batch_size, hidden_size, seq_length).
            reverse (bool): If True, apply inverse transform.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                torch.Tensor: Output tensor (batch_size, input_size, seq_length).
                torch.Tensor: Log-determinant of the Jacobian (batch_size,).
        """
        x0, x1 = torch.split(x, self.half_input_size, dim=1)  # (batch_size, half_input_size, seq_length)

        h = self.input_conv(x0)
        if g is not None:
            h = h + g

        for block in self.blocks:
            h = block(h)

        h = self.output_conv(h)  # (batch_size, half_input_size * (num_bins * 3 - 1), seq_length)

        b, _, t = h.shape
        h = h.view(b, self.half_input_size, self.num_bins * 3 - 1, t).transpose(-2, -1)  # (b, c, t, 3k - 1)
        unnorm_w = h[..., : self.num_bins] / math.sqrt(self.hidden_size)
        unnorm_h = h[..., self.num_bins : 2 * self.num_bins] / math.sqrt(self.hidden_size)
        unnorm_d = h[..., 2 * self.num_bins :]

        x1, log_abs_det = rational_quadratic_transforms(x1, unnorm_w, unnorm_h, unnorm_d, inverse=reverse, B=self.B)

        x = torch.cat([x0, x1], dim=1)  # (batch_size, input_size, frame_length)
        log_det = torch.sum(log_abs_det, dim=[1, 2])  # (batch_size,)

        return x, log_det
