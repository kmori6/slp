import copy

import torch
import torch.nn as nn

from slp.modules.transformer.types import Cache


class Decoder(nn.Module):
    """Transformer Decoder.

    Proposed in A. Vaswani et al., "Attention is all you need," in NeurIPS, 2017, pp. 5998-6008.

    """

    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])

    def forward(
        self,
        x_enc: torch.Tensor,
        x_dec: torch.Tensor,
        mask_enc: torch.Tensor,
        mask_dec: torch.Tensor,
    ) -> torch.Tensor:
        """

        Args:
            x_enc (torch.Tensor): Encoder output tensor (batch_size, source_sequence_length, d_model).
            x_dec (torch.Tensor): Decoder input tensor (batch_size, target_sequence_length, d_model).
            mask_enc (torch.Tensor): Encoder-decoder attention mask
                (batch_size, target_sequence_length, source_sequence_length).
            mask_dec (torch.Tensor): Decoder self-attention mask
                (batch_size, target_sequence_length, target_sequence_length).

        Returns:
            torch.Tensor: Output tensor (batch_size, target_sequence_length, d_model).
        """
        for layer in self.layers:
            x_dec = layer(x_enc, x_dec, mask_enc, mask_dec)

        return x_dec

    @torch.inference_mode()
    def predict(
        self, x_enc: torch.Tensor, x_dec: torch.Tensor, caches: list[Cache]
    ) -> tuple[torch.Tensor, list[Cache]]:
        """Forward pass with KV-cache for autoregressive inference.

        Args:
            x_enc (torch.Tensor): Encoder embedding tensor (1, source_sequence_length, d_model).
            x_dec (torch.Tensor): Decoder embedding tensor (1, 1, d_model).
            caches (list[Cache]): Key and value cache list for each layer (1, sequence_length - 1, d_model).

        Returns:
            tuple[torch.Tensor, list[Cache]]: Output tensor and updated caches.
        """
        if caches and len(caches) < len(self.layers):
            raise ValueError(
                f"caches must be empty or have at least {len(self.layers)} elements, but got {len(caches)}."
            )

        batch_size, target_length = x_dec.shape[:2]
        source_length = x_enc.shape[1]
        cache_length = caches[0].key.shape[1] if caches else 0

        mask_enc = torch.ones(batch_size, target_length, source_length, dtype=torch.bool, device=x_dec.device)
        mask_dec = torch.ones(
            batch_size,
            target_length,
            cache_length + target_length,
            dtype=torch.bool,
            device=x_dec.device,
        )

        caches_per_layer: list[Cache | None] = caches if caches else [None] * len(self.layers)
        new_caches = []
        for layer, cache in zip(self.layers, caches_per_layer):
            x_dec, new_cache = layer.predict(x_enc, x_dec, mask_enc, mask_dec, cache)  # type: ignore[operator]
            new_caches.append(new_cache)
        return x_dec, new_caches
