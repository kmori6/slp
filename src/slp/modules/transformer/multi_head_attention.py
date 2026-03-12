import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Multi-head attention module.

    Proposed in A. Vaswani et al., "Attention is all you need," in NeurIPS, 2017, pp. 5998-6008.

    """

    def __init__(
        self,
        hidden_size: int,
        d_k: int,
        num_heads: int,
        num_kv_heads: int,
        dropout_rate: float,
        query_bias: bool = False,
        key_bias: bool = False,
        value_bias: bool = False,
        output_bias: bool = False,
    ):
        super().__init__()
        self.h = num_heads
        self.h_kv = num_kv_heads
        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size
        self.d_k = d_k
        self.w_q = nn.Linear(hidden_size, self.h * self.d_k, bias=query_bias)
        self.w_k = nn.Linear(hidden_size, self.h_kv * self.d_k, bias=key_bias)
        self.w_v = nn.Linear(hidden_size, self.h_kv * self.d_k, bias=value_bias)
        self.w_o = nn.Linear(self.h * self.d_k, hidden_size, bias=output_bias)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            q (torch.Tensor): Query tensor (batch_size, target_sequence_length, hidden_size).
            k (torch.Tensor): Key tensor (batch_size, source_sequence_length, hidden_size).
            v (torch.Tensor): Value tensor (batch_size, source_sequence_length, hidden_size).
            mask (torch.Tensor): Mask tensor (batch_size, target_sequence_length, source_sequence_length).

        Returns:
            torch.Tensor: Output tensor (batch_size, target_sequence_length, hidden_size).
        """
        q, k, v = self._forward_qkv(q, k, v)
        x = self._forward_attention(q, k, v, mask, scale=1 / math.sqrt(self.d_k), enable_gqa=self.h_kv < self.h)
        return x

    def _forward_qkv(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Linear projection for query, key, and value.

        Args:
            q (torch.Tensor): Query tensor (batch_size, target_sequence_length, hidden_size).
            k (torch.Tensor): Key tensor (batch_size, source_sequence_length, hidden_size).
            v (torch.Tensor): Value tensor (batch_size, source_sequence_length, hidden_size).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of query, key, and value tensors
                - query: (batch_size, num_heads, target_sequence_length, d_k),
                - key: (batch_size, num_key_value_heads, source_sequence_length, d_k),
                - value: (batch_size, num_key_value_heads, source_sequence_length, d_k).
        """
        b, t, s = q.shape[0], q.shape[1], k.shape[1]
        q = self.w_q(q).view(b, t, self.h, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(b, s, self.h_kv, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(b, s, self.h_kv, self.d_k).transpose(1, 2)
        return q, k, v

    def _forward_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        scale: float,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        """Scaled dot-product attention.

        Args:
            q (torch.Tensor): Query tensor (batch_size, num_heads, target_sequence_length, d_k).
            k (torch.Tensor): Key tensor (batch_size, num_key_value_heads, source_sequence_length, d_k).
            v (torch.Tensor): Value tensor (batch_size, num_key_value_heads, source_sequence_length, d_k).
            mask (torch.Tensor): Mask tensor (batch_size, target_sequence_length, source_sequence_length).
            scale (float): Scaling factor for attention scores.
            enable_gqa (bool): Enable grouped query attention.

        Returns:
            torch.Tensor: Attention output tensor (batch_size, target_sequence_length, hidden_size).
        """
        x = nn.functional.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=mask[:, None, :, :],
            dropout_p=self.dropout_rate if self.training else 0.0,
            is_causal=False,
            scale=scale,
            enable_gqa=enable_gqa,
        )
        x = x.transpose(1, 2).flatten(2, -1)
        x = self.w_o(x)
        return x
