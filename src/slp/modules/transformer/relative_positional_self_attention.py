import math

import torch
import torch.nn as nn

from slp.modules.transformer.multi_head_attention import MultiHeadAttention


class RelativePositionalSelfAttention(MultiHeadAttention):
    """Relative positional self-attention module.

    Proposed in P. Shaw et al., "Self-attention with relative position representations," in NAACL, 2018, pp. 464-468.

    """

    def __init__(self, hidden_size: int, d_k: int, num_heads: int, dropout_rate: float, window_size: int):
        super().__init__(
            hidden_size=hidden_size,
            d_k=d_k,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            dropout_rate=dropout_rate,
            query_bias=True,
            key_bias=True,
            value_bias=True,
            output_bias=True,
        )
        self.window_size = window_size
        self.dropout = nn.Dropout(dropout_rate)
        # learned  position representations w^K_{-k...k}, w^V_{-k...k}
        self.w_rel_k = nn.Parameter(torch.Tensor(2 * window_size + 1, d_k))
        self.w_rel_v = nn.Parameter(torch.Tensor(2 * window_size + 1, d_k))
        nn.init.xavier_uniform_(self.w_rel_k)
        nn.init.xavier_uniform_(self.w_rel_v)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            q (torch.Tensor): Query tensor (batch_size, seq_length, hidden_size).
            k (torch.Tensor): Key tensor (batch_size, seq_length, hidden_size).
            v (torch.Tensor): Value tensor (batch_size, seq_length, hidden_size).
            mask (torch.Tensor): Boolean mask (batch_size, seq_length, seq_length).

        Returns:
            torch.Tensor: Output tensor (batch_size, seq_length, hidden_size).
        """
        q, k, v = self._forward_qkv(q, k, v)
        b, h, s, d_k = q.shape

        # a^K_ij = w^K_{clip(j-i, k)}, a^V_ij = w^V_{clip(j-i, k)}
        pos = torch.arange(s, device=q.device)
        distance = pos[None, :] - pos[:, None]  # (seq_length, seq_length): j - i
        rel_pos = distance.clamp(-self.window_size, self.window_size) + self.window_size  # (seq_length, seq_length)
        a_k = self.w_rel_k[rel_pos]  # (seq_length, seq_length, d_k)
        a_v = self.w_rel_v[rel_pos]  # (seq_length, seq_length, d_k)

        # e_ij = (q_i k_j^T + q_i (a^K_ij)^T) / sqrt(d_k) (5)
        qk_scores = q @ k.transpose(-2, -1)  # (batch_size, num_heads, seq_length, seq_length)
        rel_scores = (q.reshape(-1, s, d_k).transpose(0, 1) @ a_k.transpose(-2, -1)).transpose(0, 1).view(b, h, s, s)
        scores = (qk_scores + rel_scores) / math.sqrt(d_k)
        scores = scores.masked_fill(~mask[:, None, :, :], float("-inf"))

        alpha = torch.softmax(scores, dim=-1)
        alpha = self.dropout(alpha)

        # z_i = sum_j^n alpha_ij v_j + sum_j^n alpha_ij a^V_ij (3)
        v_output = alpha @ v
        rel_output = (alpha.view(-1, s, s).transpose(0, 1) @ a_v).transpose(0, 1).view(b, h, s, d_k)
        output = v_output + rel_output

        output = output.transpose(1, 2).flatten(2, -1)  # (batch, seq, hidden_size)
        output = self.w_o(output)

        return output
