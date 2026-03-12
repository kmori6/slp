import math

import torch

from slp.modules.transformer.multi_head_attention import MultiHeadAttention


def apply_rotary_embedding(
    q: torch.Tensor, k: torch.Tensor, p_q: torch.Tensor, p_k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding described in the section 3.4.2.

    Args:
        q (torch.Tensor): Query tensor (batch_size, num_heads, target_sequence_length, d_k).
        k (torch.Tensor): Key tensor (batch_size, num_key_value_heads, source_sequence_length, d_k).
        p_q (torch.Tensor): Query positional embedding tensor (1, target_sequence_length, d_k).
        p_k (torch.Tensor): Key positional embedding tensor (1, source_sequence_length, d_k).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tuple of query and key tensors
            - query: (batch_size, num_heads, target_sequence_length, d_k),
            - key: (batch_size, num_key_value_heads, source_sequence_length, d_k).
    """
    p_q = p_q[:, None, :, :]  # (1, 1, t, d_k)
    p_qsin = p_q[..., 0::2].repeat_interleave(2, dim=-1)  # (1, 1, t, d_k)
    p_qcos = p_q[..., 1::2].repeat_interleave(2, dim=-1)  # (1, 1, t, d_k)
    p_k = p_k[:, None, :, :]  # (1, 1, s, d_k)
    p_ksin = p_k[..., 0::2].repeat_interleave(2, dim=-1)  # (1, 1, s, d_k)
    p_kcos = p_k[..., 1::2].repeat_interleave(2, dim=-1)  # (1, 1, s, d_k)
    q = q * p_qcos + torch.stack([-q[..., 1::2], q[..., 0::2]], dim=-1).flatten(3, -1) * p_qsin
    k = k * p_kcos + torch.stack([-k[..., 1::2], k[..., 0::2]], dim=-1).flatten(3, -1) * p_ksin
    return q, k


class RotaryPositionalMultiHeadAttention(MultiHeadAttention):
    """Multi-head attention with rotary position embedding.

    Proposed in J. Su et al., "RoFormer: enhanced transformer with rotary position embedding,"
    in Neurocomputing, 2024, vol. 568, pp. 127063.

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
        super().__init__(
            hidden_size, d_k, num_heads, num_kv_heads, dropout_rate, query_bias, key_bias, value_bias, output_bias
        )

    def forward(  # type: ignore[override]
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        p_q: torch.Tensor,
        p_k: torch.Tensor,
    ) -> torch.Tensor:
        """

        Args:
            q (torch.Tensor): Query tensor (batch_size, target_sequence_length, hidden_size).
            k (torch.Tensor): Key tensor (batch_size, source_sequence_length, hidden_size).
            v (torch.Tensor): Value tensor (batch_size, source_sequence_length, hidden_size).c
            mask (torch.Tensor): Mask tensor (batch_size, target_sequence_length, source_sequence_length).
            p_q (torch.Tensor): Query positional embedding tensor (1, target_sequence_length, d_k).
            p_k (torch.Tensor): Key positional embedding tensor (1, source_sequence_length, d_k).

        Returns:
            torch.Tensor: Attention output tensor (batch_size, target_sequence_length, hidden_size).
        """
        q, k, v = self._forward_qkv(q, k, v)  # (b, t, h x d_k), (b, s, h x d_k), (b, s, h x d_k)
        q, k = apply_rotary_embedding(q, k, p_q, p_k)
        x = self._forward_attention(q, k, v, mask, scale=1 / math.sqrt(self.d_k), enable_gqa=self.h_kv < self.h)
        return x
