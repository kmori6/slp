import torch
import torch.nn as nn


class PredictionNetwork(nn.Module):
    """RNN Transducer module.

    Proposed in A. Graves et al., "Speech recognition with deep recrrent neural networks,"
    in ICASSP, 2013, pp. 6645-6649.

    """

    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, dropout_rate: float, blank_token_id: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, hidden_size, padding_idx=blank_token_id)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)

    def init_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float32, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=torch.float32, device=device)
        return h, c

    def forward(
        self, token: torch.Tensor, hidden_state: torch.Tensor, cell_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Args:
            token (torch.Tensor): Token tensor (batch_size, sequence_length).
            hidden_state (torch.Tensor): Hidden state tensor (num_layers, batch_size, hidden_size).
            cell_state (torch.Tensor): Cell state tensor (num_layers, batch_size, hidden_size).

        Returns:
            tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
                torch.Tensor: Hidden state tensor (batch_size, sequence_length, hidden_size).
                torch.Tensor: Hidden state tensor (num_layers, batch_size, hidden_size).
                torch.Tensor: Cell state tensor (num_layers, batch_size, hidden_size).
        """
        x = self.embed(token)  # (batch_size, sequence_length, hidden_size)
        x = self.dropout(x)
        x, (h, c) = self.lstm(x, (hidden_state, cell_state))
        return x, h, c
