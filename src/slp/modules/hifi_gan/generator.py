import math

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from slp.modules.hifi_gan.multi_receptive_field_fution import MultiReceptiveFieldFusion


class Generator(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 512,
        upsample_kernel_sizes: list[int] = [16, 16, 4, 4],
        residual_kernel_sizes: list[int] = [3, 7, 11],
        dilations_list: list[list[list[int]]] = [
            [[1, 1], [3, 1], [5, 1]],
            [[1, 1], [3, 1], [5, 1]],
            [[1, 1], [3, 1], [5, 1]],
        ],
        negative_slope: float = 0.1,
    ):
        super().__init__()
        self.conv_input = weight_norm(nn.Conv1d(input_size, hidden_size, kernel_size=7, padding=3))
        self.upsamples = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LeakyReLU(negative_slope),
                    weight_norm(
                        nn.ConvTranspose1d(
                            hidden_size // 2 ** (i - 1),
                            hidden_size // 2**i,
                            kernel_size=kernel_size,
                            stride=kernel_size // 2,
                            padding=kernel_size // 4,
                        )
                    ),
                    MultiReceptiveFieldFusion(
                        hidden_size // 2**i, negative_slope, residual_kernel_sizes, dilations_list
                    ),
                )
                for i, kernel_size in enumerate(upsample_kernel_sizes, start=1)
            ]
        )
        self.leaky_relu = nn.LeakyReLU()
        self.conv_out = weight_norm(
            nn.Conv1d(hidden_size // (2 ** len(upsample_kernel_sizes)), 1, kernel_size=7, padding=3, bias=False)
        )
        self.tanh = nn.Tanh()

    def get_upsample_size(self) -> int:
        strides = [upsample[1].stride[0] for upsample in self.upsamples]  # type: ignore[index]
        upsample_size = math.prod(strides)
        return upsample_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Mel-spectrogram tensor (batch_size, mel_size, frame_length).

        Returns:
            torch.Tensor: Waveform tensor (batch_size, 1, 256 * frame_length).
        """
        x = self.conv_input(x)  # (batch_size, hidden_size, frame_length)
        for upsample in self.upsamples:
            x = upsample(x)
        x = self.leaky_relu(x)  # (batch_size, hidden_size / 2^4, frame_length)
        x = self.conv_out(x)  # (batch_size, 1, frame_length)
        x = self.tanh(x)
        return x
