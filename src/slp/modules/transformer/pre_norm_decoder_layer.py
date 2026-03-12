import torch

from slp.modules.transformer.post_norm_decoder_layer import PostNormDecoderLayer
from slp.modules.transformer.types import Cache


class PreNormDecoderLayer(PostNormDecoderLayer):
    def forward(
        self, x_enc: torch.Tensor, x_dec: torch.Tensor, mask_enc: torch.Tensor, mask_dec: torch.Tensor
    ) -> torch.Tensor:
        """

        Args:
            x_enc (torch.Tensor): Encoder embedding sequence tensor (batch_size, source_sequence_length, d_model).
            x_dec (torch.Tensor): Decoder embedding sequence tensor (batch_size, target_sequence_length, d_model).
            mask_enc (torch.Tensor): Encoder-decoder mask tensor
                (batch_size, target_sequence_length, source_sequence_length).
            mask_dec (torch.Tensor): Decoder mask tensor (batch_size, target_sequence_length, target_sequence_length).

        Returns:
            torch.Tensor: Output sequence tensor (batch_size, target_sequence_length, d_model).
        """
        x_res = x_dec
        x_dec = self.layer_norm1(x_dec)
        x_dec = x_res + self.dropout1(self.masked_mha(x_dec, x_dec, x_dec, mask_dec))

        x_res = x_dec
        x_dec = self.layer_norm2(x_dec)
        x_dec = x_res + self.dropout2(self.mha(x_dec, x_enc, x_enc, mask_enc))

        x_res = x_dec
        x_dec = self.layer_norm3(x_dec)
        x_dec = x_res + self.dropout3(self.ffn(x_dec))

        return x_dec

    @torch.inference_mode()
    def predict(
        self,
        x_enc: torch.Tensor,
        x_dec: torch.Tensor,
        mask_enc: torch.Tensor,
        mask_dec: torch.Tensor,
        cache: Cache | None = None,
    ) -> tuple[torch.Tensor, Cache]:
        """Forward pass with KV-cache for autoregressive inference.

        Args:
            x_enc (torch.Tensor): Encoder embedding sequence tensor (batch_size, source_sequence_length, d_model).
            x_dec (torch.Tensor): Decoder input tensor (batch_size, target_sequence_length, d_model).
            mask_enc (torch.Tensor): Encoder-decoder mask tensor
                (batch_size, target_sequence_length, source_sequence_length).
            mask_dec (torch.Tensor): Decoder self-attention mask tensor
                (batch_size, target_sequence_length, cache_sequence_length + target_sequence_length).
            cache (Cache | None): Cached self-attention key/value tensors from previous decoding steps.
                When provided, each tensor has shape (batch_size, cache_sequence_length, d_model).

        Returns:
            tuple[torch.Tensor, Cache]:
                - Output tensor (batch_size, target_sequence_length, d_model).
                - Updated cache where key/value have shape
                  (batch_size, cache_sequence_length + target_sequence_length, d_model).
        """
        x_res = x_dec
        x_dec = self.layer_norm1(x_dec)
        if cache is not None:
            k = torch.cat([cache.key, x_dec], dim=1)
            v = torch.cat([cache.value, x_dec], dim=1)
        else:
            k, v = x_dec, x_dec
        new_cache = Cache(key=k, value=v)
        x_dec = x_res + self.masked_mha(x_dec, k, v, mask_dec)

        x_res = x_dec
        x_dec = self.layer_norm2(x_dec)
        x_dec = x_res + self.mha(x_dec, x_enc, x_enc, mask_enc)

        x_res = x_dec
        x_dec = self.layer_norm3(x_dec)
        x_dec = x_res + self.ffn(x_dec)

        return x_dec, new_cache
