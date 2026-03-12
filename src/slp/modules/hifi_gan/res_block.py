import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class ResBlock(nn.Module):
    def __init__(self, hidden_size: int, kernel_size: int, dilations: list[list[int]], negative_slope: float):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        self.stacks = nn.ModuleList()
        for dilation_rates in dilations:
            layers: list[nn.Module] = []
            for dilation in dilation_rates:
                layers.append(nn.LeakyReLU(negative_slope))
                layers.append(
                    weight_norm(
                        nn.Conv1d(
                            hidden_size,
                            hidden_size,
                            kernel_size,
                            stride=1,
                            padding=(dilation * (kernel_size - 1)) // 2,
                            dilation=dilation,
                        )
                    )
                )
            sequential = nn.Sequential(*layers)
            self.stacks.append(sequential)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Input tensor of shape (batch, hidden_size, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch, hidden_size, length).
        """
        for stack in self.stacks:
            x = x + stack(x)
        return x
