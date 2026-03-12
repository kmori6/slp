import torch

from slp.modules.frontend.log_mel_spectrogram import LogMelSpectrogram
from slp.utils.mask import sequence_mask


class WhisperLogMelSpectrogram(LogMelSpectrogram):
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
        x = speech.stft(
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.window_size,
            window=window,  # type: ignore[arg-type]
            center=True,
            return_complex=True,
        )  # (batch_size, freq_size, frame_length), frame_length = 1 + sequence_length // self.hop_size

        # trim the last frame
        x = x[..., :-1]  # (batch_size, freq_size, sequence_length // hop_size)
        mask = sequence_mask(length // self.hop_size).to(x.device)

        # stft -> spectrogram
        x = torch.view_as_real(x).transpose(1, 2).pow(2).sum(-1)  # (batch_size, frame_length, freq_size)

        # spectrogram -> mel-spectrogram
        x = x @ self.fbank.to(device=x.device, dtype=x.dtype)

        # mel-spectrogram -> log mel-spectrogram
        x = x.clamp(min=1e-10).log10()

        # normalization
        max_val = x.max()
        x = torch.max(x, max_val - 8.0)
        x = (x + 4.0) / 4.0

        x = x.masked_fill(~mask[:, :, None], 0.0)

        return x, mask
