import torch
import torch.nn as nn


class JointNetwork(nn.Module):
    """RNN Transducer module.

    Proposed in A. Graves et al., "Speech recognition with deep recrrent neural networks,"
    in ICASSP, 2013, pp. 6645-6649.

    """

    def __init__(self, vocab_size: int, encoder_size: int, predictor_size: int, hidden_size: int, dropout_rate: float):
        super().__init__()
        self.w_l = nn.Linear(encoder_size, hidden_size)
        self.w_p = nn.Linear(predictor_size, hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate)
        self.w_h = nn.Linear(hidden_size, vocab_size)

    def forward(self, x_enc: torch.Tensor, x_prd: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x_enc (torch.Tensor): Encoder hidden sequence tensor (batch_size, frame_length, 1, encoder_size).
            x_prd (torch.Tensor): Predicton hidden sequence tensor (batch_size, 1, sequence_length, predictor_size).

        Returns:
            torch.Tensor: Logit tesnor (batch_size, frame_length, sequence_length, vocab_size).
        """
        x = self.w_l(x_enc) + self.w_p(x_prd)  # (batch_size, frame_length, sequence_length, hidden_size)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w_h(x)  # (batch_size, frame_length, sequence_length, vocab_size)
        return x
