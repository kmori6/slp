import torch
import torch.nn as nn

from slp.modules.vits.affine_coupling_layer import AffineCouplingLayer


class Flow(nn.Module):
    """Normalizing flow module.

    Proposed in J. Kim et al., "Conditional Variational Autoencoder with Adversarial
    Learning for End-to-End Text-to-Speech," in ICML, 2021.

    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
        dilation_rate: int,
        num_blocks: int,
        num_layers: int,
        dropout_rate: float,
        cond_size: int = 0,
    ):
        super().__init__()
        self.coupling_layers = nn.ModuleList(
            [
                AffineCouplingLayer(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    num_blocks=num_blocks,
                    dropout_rate=dropout_rate,
                    cond_size=cond_size,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, g: torch.Tensor | None = None, reverse: bool = False) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Input tensor (batch_size, input_size, seq_length).
            g (torch.Tensor | None): Conditioning tensor (batch_size, cond_size, seq_length).
            reverse (bool): Whether to apply the inverse transform.

        Returns:
            torch.Tensor: Output tensor (batch_size, input_size, seq_length).
        """
        layers = self.coupling_layers if not reverse else reversed(self.coupling_layers)
        for layer in layers:
            if not reverse:
                x = layer(x, g=g, reverse=False)
                x = torch.flip(x, dims=[1])
            else:
                x = torch.flip(x, dims=[1])
                x = layer(x, g=g, reverse=True)
        return x
