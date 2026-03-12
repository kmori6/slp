import json
from pathlib import Path

import pytest
import torch
import torchaudio

from slp.dataset.speech_text_dataset import SpeechTextDataset, SpeechTextSample


@pytest.fixture
def data_file(tmp_path: Path) -> str:
    a_path = tmp_path / "a.wav"
    b_path = tmp_path / "b.wav"

    torchaudio.save(str(a_path), torch.randn(1, 160), 16_000)
    torchaudio.save(str(b_path), torch.randn(1, 80), 8_000)

    data_path = tmp_path / "data.json"
    data_path.write_text(
        json.dumps(
            [
                {"audio_path": str(a_path), "text": "hello"},
                {"audio_path": str(b_path), "text": "world"},
            ]
        ),
        encoding="utf-8",
    )
    return str(data_path)


def test_init_raises_for_non_json_file() -> None:
    with pytest.raises(ValueError, match="Data file must be a JSON file"):
        SpeechTextDataset("data.txt")


def test_len_returns_dataset_length(data_file: str) -> None:
    dataset = SpeechTextDataset(data_file)
    assert len(dataset) == 2


def test_getitem_returns_sample_without_resample(data_file: str) -> None:
    sample = SpeechTextDataset(data_file, sample_rate=16_000)[0]

    assert isinstance(sample, SpeechTextSample)
    assert sample.text == "hello"
    assert sample.speech.shape == (160,)


def test_getitem_resamples_when_sample_rate_differs(data_file: str) -> None:
    sample = SpeechTextDataset(data_file, sample_rate=16_000)[1]

    assert isinstance(sample, SpeechTextSample)
    assert sample.text == "world"
    assert sample.speech.shape == (160,)
