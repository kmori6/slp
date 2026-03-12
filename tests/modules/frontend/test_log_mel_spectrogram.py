import pytest
import torch

from slp.modules.frontend.log_mel_spectrogram import LogMelSpectrogram


@pytest.fixture
def module() -> LogMelSpectrogram:
    return LogMelSpectrogram(
        fft_size=512,
        hop_size=160,
        window_size=512,
        mel_size=80,
        sample_rate=16000,
        min_freq=0.0,
        max_freq=8000.0,
    )


def test_output_shape(module: LogMelSpectrogram) -> None:
    speech = torch.randn(2, 16000)
    length = torch.tensor([16000, 8000])

    mel, mask = module(speech, length)

    expected_frames = 16000 // module.hop_size + 1
    assert mel.shape == (2, expected_frames, 80)
    assert mask.shape == (2, expected_frames)
