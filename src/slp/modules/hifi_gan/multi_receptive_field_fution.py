import torch
import torch.nn as nn

from slp.modules.hifi_gan.res_block import ResBlock


class MultiReceptiveFieldFusion(nn.Module):
    def __init__(
        self, hidden_dim: int, negative_slope: float, kernel_sizes: list[int], dilations_list: list[list[list[int]]]
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ResBlock(hidden_dim, kernel_size, dilations, negative_slope)
                for kernel_size, dilations in zip(kernel_sizes, dilations_list)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, hidden_size, frame_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size, frame_length).
        """
        xs = [block(x) for block in self.blocks]  # list of (batch_size, hidden_size, frame_length)
        x = torch.stack(xs, dim=1)  # (batch_size, block, hidden_size, frame_length)
        x = torch.mean(x, dim=1)  # (batch_size, hidden_size, frame_length)
        return x
