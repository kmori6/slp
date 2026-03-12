import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class ScaleDiscriminator(nn.Module):
    def __init__(self, scale: int, negative_slope: float):
        super().__init__()
        self.scale = scale
        if scale > 1:
            self.avg_pool = nn.AvgPool1d(kernel_size=scale, stride=scale)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    weight_norm(nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7)),
                    nn.LeakyReLU(negative_slope),
                ),
                nn.Sequential(
                    weight_norm(nn.Conv1d(16, 64, kernel_size=41, stride=4, groups=4, padding=20)),
                    nn.LeakyReLU(negative_slope),
                ),
                nn.Sequential(
                    weight_norm(nn.Conv1d(64, 256, kernel_size=41, stride=4, groups=16, padding=20)),
                    nn.LeakyReLU(negative_slope),
                ),
                nn.Sequential(
                    weight_norm(nn.Conv1d(256, 1024, kernel_size=41, stride=4, groups=64, padding=20)),
                    nn.LeakyReLU(negative_slope),
                ),
                nn.Sequential(
                    weight_norm(nn.Conv1d(1024, 1024, kernel_size=41, stride=4, groups=256, padding=20)),
                    nn.LeakyReLU(negative_slope),
                ),
                nn.Sequential(
                    weight_norm(nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2)),
                    nn.LeakyReLU(negative_slope),
                ),
            ]
        )
        self.conv_out = weight_norm(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """

        Args:
            x (torch.Tensor): Waveform tensor (batch, 1, sample).

        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]:
                torch.Tensor: Logit tensor (batch, *).
                list[torch.Tensor]: Intermediate features (batch, *, *).
        """
        feats = []
        if self.scale > 1:
            x = self.avg_pool(x)
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        x = self.conv_out(x)
        feats.append(x)
        x = torch.flatten(x, 1, -1)
        return x, feats
