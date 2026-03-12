import math

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int, negative_slope: float):
        super().__init__()
        self.period = period
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    weight_norm(nn.Conv2d(1, 32, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
                    nn.LeakyReLU(negative_slope),
                ),
                nn.Sequential(
                    weight_norm(nn.Conv2d(32, 128, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
                    nn.LeakyReLU(negative_slope),
                ),
                nn.Sequential(
                    weight_norm(nn.Conv2d(128, 512, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
                    nn.LeakyReLU(negative_slope),
                ),
                nn.Sequential(
                    weight_norm(nn.Conv2d(512, 1024, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
                    nn.LeakyReLU(negative_slope),
                ),
            ]
        )
        self.layer = nn.Sequential(
            weight_norm(nn.Conv2d(1024, 1024, kernel_size=(5, 1), padding=(2, 0))),
            nn.LeakyReLU(negative_slope),
        )
        self.output_conv = weight_norm(nn.Conv2d(1024, 1, kernel_size=(3, 1), padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """

        Args:
            x (torch.Tensor): Waveform tensor (batch, 1, sample).

        Returns:
            tuple[torch.Tensor, list[torch.Tensor]]:
                torch.Tensor: Logit tensor (batch, *).
                list[torch.Tensor]: Intermediate features (batch, *, *, *).
        """
        # pad and reshape (batch, 1, sample) -> (batch, 1, ceil(sample / period), period)
        b, _, t = x.shape
        new_t = math.ceil(t / self.period)
        x = nn.functional.pad(x, (0, self.period * new_t - t), "reflect")
        x = x.view(b, 1, new_t, self.period)

        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        x = self.layer(x)  # (batch, 1024, *, *)
        feats.append(x)

        x = self.output_conv(x)  # (batch, 1, *, *)
        feats.append(x)

        x = torch.flatten(x, 1, -1)  # (batch, *)
        return x, feats
