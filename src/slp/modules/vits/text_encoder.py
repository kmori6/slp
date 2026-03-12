import math

import torch
import torch.nn as nn

from slp.modules.transformer.encoder import Encoder
from slp.modules.transformer.feed_forward import FeedForward
from slp.modules.transformer.post_norm_encoder_layer import PostNormEncoderLayer
from slp.modules.transformer.relative_positional_self_attention import RelativePositionalSelfAttention


class TextEncoder(nn.Module):
    """Text encoder module.

    Proposed in J. Kim et al., "Conditional Variational Autoencoder with
    Adversarial Learning for End-to-End Text-to-Speech," in ICML, 2021.

    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        ffn_size: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float,
        window_size: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        encoder_layer = PostNormEncoderLayer(
            mha_module=RelativePositionalSelfAttention(
                hidden_size=hidden_size,
                d_k=hidden_size // num_heads,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
                window_size=window_size,
            ),
            mha_norm_module=nn.LayerNorm(hidden_size),
            ffn_module=FeedForward(
                input_size=hidden_size,
                hidden_size=ffn_size,
                dropout_rate=dropout_rate,
            ),
            ffn_norm_module=nn.LayerNorm(hidden_size),
            dropout_rate=dropout_rate,
        )
        self.encoder = Encoder(encoder_layer, num_layers)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Token indices tensor (batch_size, seq_length).
            mask (torch.Tensor): Mask tensor (batch_size, seq_length).

        Returns:
            torch.Tensor: Output tensor (batch_size, seq_length, hidden_size).
        """
        h = self.embedding(x) * math.sqrt(self.hidden_size)
        h = self.encoder(h, mask[:, None, :])
        return h
