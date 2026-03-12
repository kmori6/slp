import torch
import torch.nn as nn

from slp.modules.hifi_gan.period_discriminator import PeriodDiscriminator
from slp.modules.hifi_gan.scale_discriminator import ScaleDiscriminator


class Discriminator(nn.Module):
    def __init__(
        self,
        periods: list[int] = [2, 3, 5, 7, 11],
        scales: list[int] = [1],
        negative_slope: float = 0.1,
    ):
        super().__init__()
        self.mpd = nn.ModuleList([PeriodDiscriminator(period, negative_slope=negative_slope) for period in periods])
        self.msd = nn.ModuleList([ScaleDiscriminator(scale, negative_slope=negative_slope) for scale in scales])

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        """

        Args:
            x (torch.Tensor): Waveform tensor (batch, 1, sample).

        Returns:
            tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
                list[torch.Tensor]: Logit tensors (batch, *).
                list[torch.Tensor]: Intermediate features (batch, *, *).
        """
        x_list, feats_list = [], []
        # multi-period discriminators
        for mpd in self.mpd:
            x_o, feats = mpd(x)
            x_list.append(x_o)
            feats_list.append(feats)

        # multi-scale discriminators
        for msd in self.msd:
            x_o, feats = msd(x)
            x_list.append(x_o)
            feats_list.append(feats)

        return x_list, feats_list
