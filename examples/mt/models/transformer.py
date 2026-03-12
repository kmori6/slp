import math

import torch
import torch.nn as nn

from slp.modules.transformer.decoder import Decoder
from slp.modules.transformer.encoder import Encoder
from slp.modules.transformer.feed_forward import FeedForward
from slp.modules.transformer.multi_head_attention import MultiHeadAttention
from slp.modules.transformer.positional_encoding import PositionalEncoding
from slp.modules.transformer.post_norm_decoder_layer import PostNormDecoderLayer
from slp.modules.transformer.post_norm_encoder_layer import PostNormEncoderLayer
from slp.modules.transformer.types import Cache


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout_rate: float = 0.1,
        max_length: int = 4096,
        label_smoothing: float = 0.1,
        ignore_token_id: int = -100,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by h"
        d_k = d_model // num_heads  # d_k = d_v = d_model / h (as per paper)
        self.d_model = d_model
        self.ignore_token_id = ignore_token_id

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_length)
        self.dropout = nn.Dropout(dropout_rate)

        # Encoder
        encoder_layer = PostNormEncoderLayer(
            mha_module=MultiHeadAttention(
                hidden_size=d_model,
                d_k=d_k,
                num_heads=num_heads,
                num_kv_heads=num_heads,
                dropout_rate=dropout_rate,
            ),
            mha_norm_module=nn.LayerNorm(d_model),
            ffn_module=FeedForward(
                input_size=d_model,
                hidden_size=d_ff,
                dropout_rate=dropout_rate,
                activation=nn.ReLU(),
            ),
            ffn_norm_module=nn.LayerNorm(d_model),
            dropout_rate=dropout_rate,
        )
        self.encoder = Encoder(encoder_layer, num_layers)

        # Decoder
        decoder_layer = PostNormDecoderLayer(
            masked_mha_module=MultiHeadAttention(
                hidden_size=d_model,
                d_k=d_k,
                num_heads=num_heads,
                num_kv_heads=num_heads,
                dropout_rate=dropout_rate,
            ),
            masked_mha_norm_module=nn.LayerNorm(d_model),
            mha_module=MultiHeadAttention(
                hidden_size=d_model,
                d_k=d_k,
                num_heads=num_heads,
                num_kv_heads=num_heads,
                dropout_rate=dropout_rate,
            ),
            mha_norm_module=nn.LayerNorm(d_model),
            ffn_module=FeedForward(
                input_size=d_model,
                hidden_size=d_ff,
                dropout_rate=dropout_rate,
                activation=nn.ReLU(),
            ),
            ffn_norm_module=nn.LayerNorm(d_model),
            dropout_rate=dropout_rate,
        )
        self.decoder = Decoder(decoder_layer, num_layers)

        self.linear = nn.Linear(d_model, vocab_size, bias=False)
        self.linear.weight = self.embedding.weight

        self.loss_fn = nn.CrossEntropyLoss(reduction="sum", label_smoothing=label_smoothing)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """

        Args:
            input_ids: Source sequence (batch_size, src_seq_len).
            attention_mask: Source mask (batch_size, src_seq_len).
            decoder_input_ids: Target sequence (batch_size, tgt_seq_len).
            decoder_attention_mask: Target mask (batch_size, tgt_seq_len).
            labels: Target labels (batch_size, tgt_seq_len).

        Returns:
            ModelOutput with loss and stats (loss, acc).
        """
        # Encode source sequence
        memory = self.encode(input_ids, attention_mask)
        src_attention_mask = attention_mask[:, None, :].expand(-1, input_ids.shape[1], -1).bool()

        # Decode target sequence
        decoder_attention_mask = (
            decoder_attention_mask[:, None, :].expand(-1, decoder_input_ids.shape[1], -1).tril()
        )  # (batch_size, tgt_seq_len, tgt_seq_len)
        cross_attention_mask = src_attention_mask[:, 0:1, :].expand(
            -1, decoder_input_ids.shape[1], -1
        )  # (batch_size, tgt_seq_len, src_seq_len)
        x_tgt = self.embedding(decoder_input_ids) * math.sqrt(self.d_model)
        x_tgt = x_tgt + self.positional_encoding(x_tgt)
        x_tgt = self.dropout(x_tgt)
        output = self.decoder(memory, x_tgt, cross_attention_mask, decoder_attention_mask)

        logits = self.linear(output)

        # loss
        loss = self.loss_fn(logits.flatten(0, 1), labels.flatten()) / logits.shape[0]
        valid_mask = labels != self.ignore_token_id
        acc = (logits.argmax(-1)[valid_mask] == labels[valid_mask]).sum() / valid_mask.sum()

        return {"loss": loss, "acc": acc}

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed input token IDs.

        Args:
            token_ids (torch.Tensor): Input token IDs (batch_size, seq_len).

        Returns:
            torch.Tensor: Embedded tokens (batch_size, seq_len, d_model).
        """
        x = self.embedding(token_ids) * math.sqrt(self.d_model)
        x = x + self.positional_encoding(x)
        return x

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode source sequence.

        Args:
            input_ids (torch.Tensor): Source token IDs (batch_size, src_seq_len).
            attention_mask (torch.Tensor): Source mask (batch_size, src_seq_len).

        Returns:
            torch.Tensor: Encoder output (batch_size, src_seq_len, d_model).
        """
        src_attention_mask = attention_mask[:, None, :].expand(-1, input_ids.shape[1], -1).bool()
        x_src = self.embed(input_ids)
        x_src = self.dropout(x_src)
        return self.encoder(x_src, src_attention_mask)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    @torch.inference_mode()
    def predict(
        self, x_enc: torch.Tensor, x_dec: torch.Tensor, caches: list[Cache]
    ) -> tuple[torch.Tensor, list[Cache]]:
        """Predict next token given encoder output and current decoder input.

        Args:
            x_enc (torch.Tensor): Encoder output (batch_size, src_seq_len, d_model).
            x_dec (torch.Tensor): Current decoder input (batch_size, tgt_seq_len, d_model).
            caches (list[Cache]): List of decoder layer caches for auto-regressive decoding.

        Returns:
            tuple[torch.Tensor, list[Cache]]: Tuple of (logits for next token, updated caches).
        """
        return self.decoder.predict(x_enc, x_dec, caches)
