import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from slp.collate_fn.speech_to_text import SpeechToTextCollateFn
from slp.dataset.speech_text_dataset import SpeechTextSample
from slp.modules.frontend.log_mel_spectrogram import LogMelSpectrogram


@pytest.fixture
def collate_fn() -> SpeechToTextCollateFn:
    vocab = {"[BLANK]": 0, "[UNK]": 1, "[PAD]": 2, "hello": 3, "world": 4, "hoge": 5}
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]")
    frontend = LogMelSpectrogram(
        fft_size=512,
        hop_size=160,
        window_size=400,
        mel_size=80,
        sample_rate=16_000,
        min_freq=0.0,
        max_freq=8_000.0,
    )
    return SpeechToTextCollateFn(tokenizer, frontend, max_length=128)


def test_collate_fn(collate_fn: SpeechToTextCollateFn):
    sample_list = [
        SpeechTextSample(speech=torch.randn(16_000), text="hello world"),
        SpeechTextSample(speech=torch.randn(8_000), text="hoge"),
    ]

    batch = collate_fn(sample_list)

    assert set(batch.keys()) == {"input_values", "attention_mask", "labels"}
    assert batch["input_values"].shape == (2, 101, 80)  # (batch_size, frame_length, mel_size)
    assert batch["attention_mask"].shape == (2, 101)  # (batch_size, frame_length)
    assert batch["labels"].shape == (2, 2)  # (batch_size, sequence_length)
    assert batch["labels"][1, 1] == -100
