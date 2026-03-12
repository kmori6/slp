import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from slp.collate_fn.text_to_speech import TextToSpeechCollateFn
from slp.dataset.speech_text_dataset import SpeechTextSample
from slp.modules.frontend.linear_spectrogram import LinearSpectrogram


@pytest.fixture
def collate_fn() -> TextToSpeechCollateFn:
    vocab = {"[BLANK]": 0, "[UNK]": 1, "[PAD]": 2, "hello": 3, "world": 4, "hoge": 5}
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]")
    frontend = LinearSpectrogram(fft_size=512, hop_size=160, window_size=400)
    return TextToSpeechCollateFn(
        tokenizer=tokenizer,
        frontend=frontend,
        max_length=128,
        segment_size=32,
        pad_token_id=tokenizer.pad_token_id,
    )


def test_collate_fn(collate_fn: TextToSpeechCollateFn):
    sample_list = [
        SpeechTextSample(speech=torch.randn(16_000), text="hello world"),
        SpeechTextSample(speech=torch.randn(8_000), text="hoge"),
    ]

    batch = collate_fn(sample_list)

    assert set(batch.keys()) == {
        "input_ids",
        "attention_mask",
        "speech",
        "xlin",
        "xlin_mask",
        "start_indices",
    }
    assert batch["input_ids"].shape == (2, 2)  # (batch_size, text_length)
    assert batch["attention_mask"].shape == (2, 2)  # (batch_size, text_length)
    assert batch["speech"].shape == (2, 16_000)  # (batch_size, sample_length)
    assert batch["xlin"].shape == (2, 257, 101)  # (batch_size, freq_size, frame_length)
    assert batch["xlin_mask"].shape == (2, 101)  # (batch_size, frame_length)
    assert batch["xlin_mask"][0].sum().item() == 101
    assert batch["xlin_mask"][1].sum().item() == 51
    assert batch["xlin"][1, :, 51:].abs().sum().item() == 0.0

    xlin_length = batch["xlin_mask"].sum(dim=-1)
    assert batch["start_indices"].shape == (2,)
    assert batch["start_indices"].dtype == torch.long
    assert torch.all(batch["start_indices"] >= 0)
    assert torch.all(batch["start_indices"] <= (xlin_length - collate_fn.segment_size).clamp(min=0))
