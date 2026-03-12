from enum import Enum

import torch
import torch.nn as nn
from torchaudio.functional import melscale_fbanks

from slp.utils.mask import sequence_mask


class SpectrogramType(Enum):
    POWER = "power"
    MAGNITUDE = "magnitude"


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        fft_size: int,
        hop_size: int,
        window_size: int,
        mel_size: int,
        sample_rate: int,
        min_freq: float,
        max_freq: float,
        spec_type: SpectrogramType = SpectrogramType.POWER,
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.window_size = window_size
        window = torch.hann_window(window_size, dtype=torch.float32)
        self.register_buffer("window", window)
        fbank = melscale_fbanks(
            n_freqs=fft_size // 2 + 1,
            f_min=min_freq,
            f_max=max_freq,
            n_mels=mel_size,
            sample_rate=sample_rate,
            # NOTE: parameters for librosa.filters.mel of htk=False and norm="slaney"
            norm="slaney",
            mel_scale="slaney",
        )
        self.register_buffer("fbank", fbank)  # (freq_size, mel_size)
        self.spec_type = spec_type

    def forward(self, speech: torch.Tensor, length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            speech (torch.Tensor): Speech tensor (batch_size, sequence_length).
            length (torch.Tensor): Speech length tensor (batch_size,).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                torch.Tensor: Log mel-spectrogram tensor (batch_size, sequence_length // hop_size + 1, mel_size).
                torch.Tensor: Mask tensor (batch_size, sequence_length // hop_size + 1).
        """
        # signal -> stft
        window = self.window.to(dtype=speech.dtype, device=speech.device)
        # frame_length = 1 + seq_len // hop_size
        x = speech.stft(
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.window_size,
            window=window,
            center=True,
            return_complex=True,
        )  # (batch_size, freq_size, frame_length)
        mask = sequence_mask(1 + length // self.hop_size).to(x.device)

        # stft -> spectrogram
        x = torch.view_as_real(x).transpose(1, 2).pow(2).sum(-1)  # (batch_size, frame_length, freq_size)
        if self.spec_type == SpectrogramType.MAGNITUDE:
            x = x.sqrt()

        # spectrogram -> mel-spectrogram
        x = x @ self.fbank.to(device=x.device, dtype=x.dtype)

        # mel-spectrogram -> log mel-spectrogram
        x = x.clamp(min=1e-10).log()
        x = x.masked_fill(~mask[:, :, None], 0.0)

        return x, mask
