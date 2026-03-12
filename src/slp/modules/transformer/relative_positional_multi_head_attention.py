import math

import torch
import torch.nn as nn

from slp.modules.transformer.multi_head_attention import MultiHeadAttention
from slp.modules.transformer.positional_encoding import PositionalEncoding


def left_shift(x: torch.Tensor):
    """Shift positional tensor for relative positional calculation.

    Args:
        x (torch.Tensor): Embedding tensor (*, L, M + L).

    Returns:
        torch.Tensor: Left-shifted tensor (*, L, M).

    Examples:
        >>> M = L = 5
        >>> x = torch.arange(M + L - 1, -1, -1).repeat(1, L, 1) + 10
        >>> x
        tensor([[[19, 18, 17, 16, 15, 14, 13, 12, 11, 10],
                 [19, 18, 17, 16, 15, 14, 13, 12, 11, 10],
                 [19, 18, 17, 16, 15, 14, 13, 12, 11, 10],
                 [19, 18, 17, 16, 15, 14, 13, 12, 11, 10],
                 [19, 18, 17, 16, 15, 14, 13, 12, 11, 10]]])
        >>> left_shift(x)
        tensor([[[10,  0,  0,  0,  0],
                 [11, 10,  0,  0,  0],
                 [12, 11, 10,  0,  0],
                 [13, 12, 11, 10,  0],
                 [14, 13, 12, 11, 10]]])
    """
    L = x.shape[-2]
    M = x.shape[-1] - L
    assert M > 0
    return x.flatten(-2, -1)[..., 2 * L - 1 :].unfold(-1, size=M, step=M + L - 1).tril(M - L)  # (*, L, M)


class RelativePositionalMultiHeadAttention(MultiHeadAttention):
    """Multi-head attention with relative positional encoding.

    Proposed in Z. Dai et al., "Transformer-XL: attentive language models beyond a fixed-length context,"
    in ACL, 2019, pp. 2978-2988.

    """

    def __init__(self, hidden_size: int, d_k: int, num_heads: int, dropout_rate: float):
        super().__init__(hidden_size, d_k, num_heads, num_heads, dropout_rate)
        self.pe = PositionalEncoding(hidden_size)
        self.w_p = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_u = nn.Parameter(torch.empty(self.h, self.d_k), requires_grad=True)
        self.b_v = nn.Parameter(torch.empty(self.h, self.d_k), requires_grad=True)
        # NOTE: initialize parameters same as "Linear"
        # https://pytorch.org/docs/main/generated/torch.nn.Linear.html#torch.nn.Linear
        nn.init.uniform_(self.b_u, -math.sqrt(1 / hidden_size), math.sqrt(1 / hidden_size))
        nn.init.uniform_(self.b_v, -math.sqrt(1 / hidden_size), math.sqrt(1 / hidden_size))

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            q (torch.Tensor): Query tensor (batch_size, target_sequence_length, hidden_size).
            k (torch.Tensor): Key tensor (batch_size, source_sequence_length, hidden_size).
            v (torch.Tensor): Value tensor (batch_size, source_sequence_length, hidden_size).
            mask (torch.Tensor): Mask tensor (batch_size, target_sequence_length, source_sequence_length).

        Returns:
            torch.Tensor: Attention output tensor (batch_size, target_sequence_length, hidden_size).
        """
        q, k, v = self._forward_qkv(q, k, v)  # (b, h, t, d_k), (b, h, s, d_k), (b, h, s, d_k)
        s = k.shape[2]

        # positional encoding where t = 2s - 1, ..., 0 (reverse order)
        p = self.pe(k.new_ones(1, 2 * s, 1)).flip(1)  # (1, 2s, h x d_k)
        p = self.w_p(p).view(1, 2 * s, self.h, self.d_k).transpose(1, 2)  # (1, h, 2s, d_k)

        # attention score in section 3.3 and appendix B
        ac = torch.matmul(q + self.b_u[None, :, None, :], k.transpose(2, 3))  # (b, h, t, s)
        bd = torch.matmul(q + self.b_v[None, :, None, :], p.transpose(2, 3))  # (b, h, t, 2s)
        bd = left_shift(bd)  # (b, h, t, s)
        x = (ac + bd) / math.sqrt(self.d_k)  # (b, h, t, s)

        x = x.masked_fill(~mask[:, None, :, :], float("-inf"))
        x = torch.softmax(x, dim=-1).masked_fill(~mask[:, None, :, :], 0.0)
        x = torch.matmul(x, v)  # (b, h, t, d_k)
        x = x.transpose(1, 2).flatten(2, -1)  # (b, t, h x d_k)
        x = self.w_o(x)
        return x
