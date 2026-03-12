import torch
import torch.nn as nn

from slp.utils.mask import sequence_mask


class LinearSpectrogram(nn.Module):
    def __init__(self, fft_size: int, hop_size: int, window_size: int):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.window_size = window_size
        window = torch.hann_window(window_size, dtype=torch.float32)
        self.register_buffer("window", window)

    def forward(self, speech: torch.Tensor, length: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            speech (torch.Tensor): Speech tensor (batch_size, sequence_length).
            length (torch.Tensor): Speech length tensor (batch_size,).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                torch.Tensor: Linear spectrogram tensor (batch_size, freq_size, frame_length).
                torch.Tensor: Mask tensor (batch_size, frame_length).
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
        x = torch.view_as_real(x).pow(2).sum(-1).sqrt()
        x = x.masked_fill(~mask[:, None, :], 0.0)

        return x, mask
