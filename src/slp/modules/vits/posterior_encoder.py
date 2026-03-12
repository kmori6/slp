import torch
import torch.nn as nn

from slp.modules.wavenet.residual_block import ResidualBlock


class PosteriorEncoder(nn.Module):
    """Posterior encoder module.

    Proposed in J. Kim et al., "Conditional Variational Autoencoder with
    Adversarial Learning for End-to-End Text-to-Speech," in ICML, 2021.

    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        kernel_size: int,
        dilation_rate: int,
        num_layers: int,
        dropout_rate: float,
        cond_size: int = 0,
    ):
        super().__init__()
        self.output_size = output_size

        self.input_conv = nn.Conv1d(input_size, hidden_size, 1)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    input_size=hidden_size,
                    kernel_size=kernel_size,
                    dilation=dilation_rate**i,
                    dropout_rate=dropout_rate,
                    cond_size=cond_size,
                )
                for i in range(num_layers)
            ]
        )
        self.output_conv = nn.Conv1d(hidden_size, output_size * 2, 1)

    def forward(
        self, x: torch.Tensor, g: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Args:
            x (torch.Tensor): Input tensor (batch_size, input_size, frame_length).
            g (torch.Tensor | None): Global conditioning tensor (batch_size, cond_size, frame_length).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                z (torch.Tensor): Sampled latent tensor (batch_size, output_size, frame_length).
                mean (torch.Tensor): Mean of the posterior (batch_size, output_size, frame_length).
                log_std (torch.Tensor): Log standard deviation (batch_size, output_size, frame_length).
        """
        x = self.input_conv(x)  # (batch_size, hidden_size, frame_length)

        skip_sum = torch.zeros_like(x)
        for block in self.blocks:
            x, skip = block(x, g)
            skip_sum = skip_sum + skip

        stats = self.output_conv(skip_sum)  # (batch_size, output_size * 2, frame_length)

        mean, log_std = torch.split(stats, self.output_size, dim=1)  # 2 * (batch_size, output_size, frame_length)
        z = mean + torch.randn_like(mean) * torch.exp(log_std)

        return z, mean, log_std
