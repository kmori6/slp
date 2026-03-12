import pytest
import torch

from slp.modules.frontend.linear_spectrogram import LinearSpectrogram


@pytest.fixture
def module() -> LinearSpectrogram:
    return LinearSpectrogram(
        fft_size=512,
        hop_size=160,
        window_size=512,
    )


def test_output_shape(module: LinearSpectrogram) -> None:
    speech = torch.randn(2, 16000)
    length = torch.tensor([16000, 8000])

    spec, mask = module(speech, length)

    expected_frames = 16000 // module.hop_size + 1
    expected_freq_bins = module.fft_size // 2 + 1
    assert spec.shape == (2, expected_freq_bins, expected_frames)
    assert mask.shape == (2, expected_frames)
