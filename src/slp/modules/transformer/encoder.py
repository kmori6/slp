import copy
from typing import Any

import torch
import torch.nn as nn

from slp.modules.transformer.types import Cache


class Encoder(nn.Module):
    """Transformer Encoder.

    Proposed in A. Vaswani et al., "Attention is all you need," in NeurIPS, 2017, pp. 5998-6008.

    """

    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, mask: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Input tensor (batch_size, sequence_length, d_model).
            mask (torch.Tensor): Attention mask (batch_size, sequence_length, sequence_length).
            **kwargs: Additional arguments passed to each layer (e.g. p_q, p_k).

        Returns:
            torch.Tensor: Output tensor (batch_size, sequence_length, d_model).
        """
        for layer in self.layers:
            x = layer(x, mask, **kwargs)

        return x

    @torch.inference_mode()
    def predict(
        self, x: torch.Tensor, mask: torch.Tensor, caches: list[Cache], **kwargs: Any
    ) -> tuple[torch.Tensor, list[Cache]]:
        """Forward pass with KV-cache for autoregressive inference.

        Args:
            x (torch.Tensor): Input tensor (batch_size, sequence_length, d_model).
            mask (torch.Tensor): Attention mask (batch_size, sequence_length, sequence_length).
            caches (list[Cache]): KV-caches from previous step (empty list for first step).
            **kwargs: Additional arguments passed to each layer (e.g. p_q, p_k).

        Returns:
            tuple[torch.Tensor, list[Cache]]: Output tensor and updated caches.
        """
        caches_per_layer: list[Cache | None] = caches if caches else [None] * len(self.layers)
        new_caches = []
        for layer, cache in zip(self.layers, caches_per_layer):
            x, new_cache = layer.predict(x, mask, cache, **kwargs)  # type: ignore[operator]
            new_caches.append(new_cache)
        return x, new_caches
