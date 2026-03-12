import torch
import torch.nn as nn

from slp.modules.frontend.log_mel_spectrogram import LogMelSpectrogram, SpectrogramType


class MelSpectrogramLoss(nn.Module):
    def __init__(
        self,
        fft_size: int,
        hop_size: int,
        window_size: int,
        mel_size: int,
        sample_rate: int,
        min_freq: float,
        max_freq: float,
    ):
        super().__init__()
        self.mel_spectrogram = LogMelSpectrogram(
            fft_size=fft_size,
            hop_size=hop_size,
            window_size=window_size,
            mel_size=mel_size,
            sample_rate=sample_rate,
            min_freq=min_freq,
            max_freq=max_freq,
            spec_type=SpectrogramType.MAGNITUDE,
        )
        self.l1_loss_fn = nn.L1Loss(reduction="mean")

    def forward(self, fake_wav: torch.Tensor, real_wav: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        """

        Args:
            fake_wav (torch.Tensor): Waveform tensor from the generator (batch, sample).
            real_wav (torch.Tensor): Waveform tensor from the ground truth audio (batch, sample).
            length (torch.Tensor): Waveform length tensor (batch).

        Returns:
            torch.Tensor: Mel-spectrogram loss.
        """
        if fake_wav.shape != real_wav.shape:
            raise ValueError(f"Shape mismatch between fake_wav and real_wav: {fake_wav.shape} vs {real_wav.shape}")
        fake_mel, _ = self.mel_spectrogram(fake_wav, length)  # (batch, frame, mel_size)
        real_mel, _ = self.mel_spectrogram(real_wav, length)  # (batch, frame, mel_size)
        loss = self.l1_loss_fn(fake_mel, real_mel)
        return loss
