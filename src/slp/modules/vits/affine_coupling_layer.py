import torch
import torch.nn as nn

from slp.modules.wavenet.residual_block import ResidualBlock


class AffineCouplingLayer(nn.Module):
    """Volume-preserving affine coupling layer with a stack of WaveNet residual blocks.

    Refer to L. Dinh et al., "Density estimation using Real NVP," in ICLR, 2017.

    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
        dilation_rate: int,
        num_blocks: int,
        dropout_rate: float,
        cond_size: int = 0,
    ):
        super().__init__()
        assert input_size % 2 == 0
        self.half_input_size = input_size // 2
        self.input_conv = nn.Conv1d(self.half_input_size, hidden_size, 1)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    input_size=hidden_size,
                    kernel_size=kernel_size,
                    dilation=dilation_rate**i,
                    dropout_rate=dropout_rate,
                    cond_size=cond_size,
                )
                for i in range(num_blocks)
            ]
        )
        self.output_conv = nn.Conv1d(hidden_size, self.half_input_size, 1)

        # NOTE: Zero-initialized so the layer starts as an identity transform (Glow; Kingma & Dhariwal, 2018).
        nn.init.zeros_(self.output_conv.weight)
        if self.output_conv.bias is not None:
            nn.init.zeros_(self.output_conv.bias)

    def forward(self, x: torch.Tensor, g: torch.Tensor | None = None, reverse: bool = False) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Input tensor (batch_size, input_size, seq_length).
            g (torch.Tensor | None): Conditioning tensor (batch_size, cond_size, seq_length).
            reverse (bool): Whether to apply the inverse transform.

        Returns:
            torch.Tensor: Output tensor (batch_size, input_size, seq_length).
        """
        x0, x1 = torch.split(x, self.half_input_size, dim=1)  # 2 * (batch_size, input_size / 2, seq_length)

        h = self.input_conv(x0)  # (batch_size, hidden_size, seq_length)

        skip_sum = torch.zeros_like(h)
        for block in self.blocks:
            h, skip = block(h, g=g)
            skip_sum = skip_sum + skip

        m = self.output_conv(skip_sum)  # (batch_size, input_size / 2, seq_length)

        if not reverse:
            x1 = m + x1
        else:
            x1 = x1 - m

        return torch.cat([x0, x1], dim=1)  # (batch_size, input_size, seq_length)
