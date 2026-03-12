from typing import Any

import torch

from slp.modules.transformer.post_norm_encoder_layer import PostNormEncoderLayer
from slp.modules.transformer.types import Cache


class PreNormEncoderLayer(PostNormEncoderLayer):
    def forward(self, x: torch.Tensor, mask: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Embedding tensor (batch_size, sequence_length, d_model).
            mask (torch.Tensor): Mask tensor (batch_size, sequence_length, sequence_length).
            **kwargs: Additional arguments passed to the MHA module (e.g. p_q, p_k).

        Returns:
            torch.Tensor: Output tensor (batch_size, sequence_length, d_model).
        """
        x_res = x
        x = self.layer_norm1(x)
        x = x_res + self.dropout1(self.mha(x, x, x, mask, **kwargs))

        x_res = x
        x = self.layer_norm2(x)
        x = x_res + self.dropout2(self.ffn(x))

        return x

    @torch.inference_mode()
    def predict(
        self, x: torch.Tensor, mask: torch.Tensor, cache: Cache | None, **kwargs: Any
    ) -> tuple[torch.Tensor, Cache]:
        """Forward pass with KV-cache for autoregressive inference.

        Args:
            x (torch.Tensor): Input tensor (batch_size, sequence_length, d_model).
            mask (torch.Tensor): Mask tensor (batch_size, sequence_length, sequence_length).
            cache (Cache | None): KV-cache from previous step.
            **kwargs: Additional arguments passed to the MHA module (e.g. p_q, p_k).

        Returns:
            tuple[torch.Tensor, Cache]: Output tensor and updated cache.
        """
        x_res = x
        x = self.layer_norm1(x)
        if cache is not None:
            k = torch.cat([cache.key, x], dim=1)
            v = torch.cat([cache.value, x], dim=1)
        else:
            k, v = x, x
        new_cache = Cache(key=k, value=v)
        x = x_res + self.mha(x, k, v, mask, **kwargs)

        x_res = x
        x = self.layer_norm2(x)
        x = x_res + self.ffn(x)

        return x, new_cache
