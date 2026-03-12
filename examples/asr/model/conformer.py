import torch
import torch.nn as nn
from torchaudio.functional import rnnt_loss

from slp.modules.conformer.streaming_encoder import StreamingEncoder
from slp.modules.rnnt.joint_network import JointNetwork
from slp.modules.rnnt.prediction_network import PredictionNetwork


class ConformerRNNT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        input_size: int,
        hidden_size: int,
        num_heads: int,
        kernel_size: int,
        num_blocks: int,
        num_layers: int,
        pred_size: int,
        joint_size: int,
        dropout_rate: float,
        blank_token_id: int,
        ignore_token_id: int,
        min_chunk_size: int,
        max_chunk_size: int,
        ctc_loss_weight: float,
        streaming_mask_ratio: float = 0.6,
    ):
        super().__init__()
        self.blank_token_id = blank_token_id
        self.ignore_token_id = ignore_token_id
        self.ctc_loss_weight = ctc_loss_weight
        self.encoder = StreamingEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            kernel_size=kernel_size,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
            streaming_mask_ratio=streaming_mask_ratio,
        )
        self.prediction_network = PredictionNetwork(
            vocab_size=vocab_size,
            hidden_size=pred_size,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            blank_token_id=blank_token_id,
        )
        self.joint_network = JointNetwork(
            vocab_size=vocab_size,
            encoder_size=hidden_size,
            predictor_size=pred_size,
            hidden_size=joint_size,
            dropout_rate=dropout_rate,
        )
        self.ctc_linear = nn.Linear(hidden_size, vocab_size)
        self.ctc_loss_fn = nn.CTCLoss(blank=blank_token_id, zero_infinity=True, reduction="sum")

    def forward(
        self, input_values: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """

        Args:
            input_values (torch.Tensor): Acoustic feature tensor (batch_size, frame_length, input_size).
            attention_mask (torch.Tensor): Mask tensor (batch_size, frame_length).
            labels (torch.Tensor): Target labels (batch_size, sequence_length).

        Returns:
            dict[str, torch.Tensor]:
                - loss (torch.Tensor): Total loss (scalar).
                - rnnt_loss (torch.Tensor): RNN-T loss (scalar).
                - ctc_loss (torch.Tensor): CTC loss (scalar).
        """

        # encoder
        x, mask = self.encode(input_values, attention_mask)  # (batch_size, frame_length', encoder_size)

        # prediction network
        batch_size = input_values.shape[0]
        input_ids = labels.detach().clone()
        input_ids[input_ids == self.ignore_token_id] = self.blank_token_id
        input_ids = torch.cat(
            [torch.full((batch_size, 1), self.blank_token_id, device=input_ids.device), input_ids], dim=1
        )  # (batch_size, sequence_length + 1)
        (h, c) = self.prediction_network.init_state(batch_size, x.device)
        y, _, _ = self.prediction_network(input_ids, h, c)  # (batch_size, sequence_length, predictor_size)

        # joint network
        logits = self.joint_network(
            x[:, :, None, :], y[:, None, :, :]
        )  # (batch_size, frame_length, sequence_length, vocab_size)

        # loss
        rnnt_loss = self.rnnt_loss(logits, mask, labels)
        ctc_loss = self.ctc_loss(x, mask, labels)
        loss = rnnt_loss + self.ctc_loss_weight * ctc_loss

        return {"loss": loss, "rnnt_loss": rnnt_loss, "ctc_loss": ctc_loss}

    def rnnt_loss(self, logits: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute RNN-T loss.

        Args:
            logits (torch.Tensor): Joint network output tensor (batch_size, frame_length', sequence_length, vocab_size).
            mask (torch.Tensor): Mask tensor (batch_size, frame_length').
            labels (torch.Tensor): Target labels (batch_size, sequence_length).

        Returns:
            torch.Tensor: RNN-T loss (scalar).
        """
        targets = labels.masked_fill(
            labels == self.ignore_token_id, self.blank_token_id
        )  # (batch_size, sequence_length)
        input_lengths = mask.sum(-1)  # (batch_size,)
        target_lengths = (labels != self.ignore_token_id).sum(-1)  # (batch_size,)
        return rnnt_loss(
            logits.float(),
            targets.int(),
            input_lengths.int(),
            target_lengths.int(),
            blank=self.blank_token_id,
            reduction="mean",
            fused_log_softmax=True,
        )

    def ctc_loss(self, x: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute CTC loss.

        Args:
            x (torch.Tensor): Encoder output tensor (batch_size, frame_length', hidden_size).
            mask (torch.Tensor): Mask tensor (batch_size, frame_length').
            labels (torch.Tensor): Target labels (batch_size, sequence_length), padded with ignore_token_id.

        Returns:
            torch.Tensor: CTC loss (scalar).
        """
        log_probs = self.ctc_linear(x).log_softmax(dim=-1)  # (batch_size, frame_length', vocab_size)
        log_probs = log_probs.transpose(0, 1)  # (frame_length', batch_size, vocab_size)
        input_lengths = mask.sum(-1)  # (batch_size,)
        targets = labels.masked_fill(
            labels == self.ignore_token_id, self.blank_token_id
        )  # (batch_size, sequence_length)
        target_lengths = (labels != self.ignore_token_id).sum(-1)  # (batch_size,)
        loss = self.ctc_loss_fn(log_probs, targets.int(), input_lengths.int(), target_lengths.int())
        return loss / targets.shape[0]  # average over batch

    def encode(self, input_values: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input acoustic features.

        Args:
            input_values (torch.Tensor): Acoustic feature tensor (batch_size, frame_length, input_size).
            attention_mask (torch.Tensor): Padding mask tensor (batch_size, frame_length).

        Return:
            tuple[torch.Tensor, torch.Tensor]: Encoder output and mask
                (batch_size, frame_length', encoder_size), (batch_size, frame_length').
        """
        # encoder
        input_lengths = attention_mask.sum(-1)
        return self.encoder(input_values, input_lengths)  # (batch_size, frame_length', encoder_size)
