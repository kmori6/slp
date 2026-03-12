import pytest
import torch
from transformers import WhisperProcessor

from slp.modules.frontend.whisper_log_mel_spectrogram import WhisperLogMelSpectrogram


@pytest.fixture
def module() -> WhisperLogMelSpectrogram:
    return WhisperLogMelSpectrogram(
        fft_size=400,
        hop_size=160,
        window_size=400,
        mel_size=128,
        sample_rate=16000,
        min_freq=0.0,
        max_freq=8000.0,
    )


def test_output_shape(module: WhisperLogMelSpectrogram):
    speech = torch.randn(2, 16000)
    length = torch.tensor([16000, 8000])

    mel, mask = module(speech, length)

    # Whisper trims the last frame: seq_len // hop_size (not +1)
    expected_frames = 16000 // module.hop_size
    assert mel.shape == (2, expected_frames, 128)
    assert mask.shape == (2, expected_frames)


def test_frontend(
    module: WhisperLogMelSpectrogram,
    model_name: str = "openai/whisper-large-v3-turbo",
    rtol: float = 1e-4,
    atol: float = 1e-4,
):
    processor = WhisperProcessor.from_pretrained(model_name)
    length = 16000 * 30
    speech = torch.randn(length)

    outputs = processor(speech, sampling_rate=16000, return_tensors="pt")
    x, _ = module(speech[None, :], torch.tensor([length]))

    torch.testing.assert_close(x.transpose(1, 2), outputs.input_features, rtol=rtol, atol=atol)
