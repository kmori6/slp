import torch
import torch.nn as nn

from slp.modules.vits.dds_conv import DDSConv


class ConditionEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
        num_layers: int,
        dropout_rate: float,
        cond_size: int = 0,
    ):
        super().__init__()
        self.input_conv = nn.Conv1d(input_size, hidden_size, 1)
        self.blocks = nn.ModuleList(
            [
                DDSConv(
                    hidden_size=hidden_size,
                    kernel_size=kernel_size,
                    dilation=kernel_size**i,
                    dropout_rate=dropout_rate,
                )
                for i in range(num_layers)
            ]
        )
        self.output_conv = nn.Conv1d(hidden_size, hidden_size, 1)

        if cond_size > 0:
            self.cond_conv = nn.Conv1d(cond_size, hidden_size, 1)

    def forward(self, x: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Input tensor (batch_size, input_size, seq_length).
            g (torch.Tensor | None): Global conditioning tensor (batch_size, cond_size, seq_length).

        Returns:
            torch.Tensor: Output tensor (batch_size, hidden_size, seq_length).
        """
        x = self.input_conv(x)

        if g is not None:
            x = x + self.cond_conv(g)

        for block in self.blocks:
            x = block(x)

        x = self.output_conv(x)

        return x
