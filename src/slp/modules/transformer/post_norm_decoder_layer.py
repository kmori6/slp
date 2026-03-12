import torch
import torch.nn as nn

from slp.modules.transformer.types import Cache


class PostNormDecoderLayer(nn.Module):
    def __init__(
        self,
        masked_mha_module: nn.Module,
        masked_mha_norm_module: nn.Module,
        mha_module: nn.Module,
        mha_norm_module: nn.Module,
        ffn_module: nn.Module,
        ffn_norm_module: nn.Module,
        dropout_rate: float,
    ):
        super().__init__()
        self.masked_mha = masked_mha_module
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer_norm1 = masked_mha_norm_module
        self.mha = mha_module
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer_norm2 = mha_norm_module
        self.ffn = ffn_module
        self.dropout3 = nn.Dropout(dropout_rate)
        self.layer_norm3 = ffn_norm_module

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
        x_dec = self.layer_norm1(x_dec + self.dropout1(self.masked_mha(x_dec, x_dec, x_dec, mask_dec)))
        x_dec = self.layer_norm2(x_dec + self.dropout2(self.mha(x_dec, x_enc, x_enc, mask_enc)))
        x_dec = self.layer_norm3(x_dec + self.dropout3(self.ffn(x_dec)))
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
        """

        Args:
            x (torch.Tensor): Embedding tensor (batch_size, 1, d_model).
            mask (torch.Tensor): Mask tensor (1, 1, sequence_length).
            cache (Cache): Key and value cache. (1, sequence_length - 1, d_model).

        Returns:
            torch.Tensor: Output tensor (batch_size, 1, d_model).
        """
        r = x_dec
        if cache is not None:
            k = torch.cat([cache.key, x_dec], dim=1)  # (1, sequence_length, d_model)
            v = torch.cat([cache.value, x_dec], dim=1)  # (1, sequence_length, d_model)
        else:
            k, v = x_dec, x_dec
        x_dec = self.masked_mha(x_dec, k, v, mask_dec)
        x_dec = self.layer_norm1(r + x_dec)
        x_dec = self.layer_norm2(x_dec + self.mha(x_dec, x_enc, x_enc, mask_enc))
        x_dec = self.layer_norm3(x_dec + self.ffn(x_dec))
        return x_dec, Cache(key=k, value=v)
