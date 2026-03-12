import torch
import torch.nn as nn


class PostNormEncoderLayer(nn.Module):
    def __init__(
        self,
        mha_module: nn.Module,
        mha_norm_module: nn.Module,
        ffn_module: nn.Module,
        ffn_norm_module: nn.Module,
        dropout_rate: float,
    ):
        super().__init__()
        self.mha = mha_module
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer_norm1 = mha_norm_module
        self.ffn = ffn_module
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer_norm2 = ffn_norm_module

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Embedding tensor (batch_size, sequence_length, d_model).
            mask (torch.Tensor): Mask tensor (batch_size, sequence_length, sequence_length).

        Returns:
            torch.Tensor: Output tensor (batch_size, sequence_length, d_model).
        """
        x = self.layer_norm1(x + self.dropout1(self.mha(x, x, x, mask)))
        x = self.layer_norm2(x + self.dropout2(self.ffn(x)))
        return x
