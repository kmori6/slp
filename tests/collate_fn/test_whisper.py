import pytest
import torch
from transformers import WhisperFeatureExtractor, WhisperTokenizer

from slp.collate_fn.whisper import WhisperCollateFn
from slp.dataset.speech_text_dataset import SpeechTextSample


@pytest.fixture
def collate_fn() -> WhisperCollateFn:
    model_name = "openai/whisper-large-v3-turbo"
    tokenizer = WhisperTokenizer.from_pretrained(model_name)
    frontend = WhisperFeatureExtractor.from_pretrained(model_name)
    return WhisperCollateFn(
        tokenizer=tokenizer,
        frontend=frontend,
        max_length=128,
        pad_token_id=tokenizer.pad_token_id,
        ignore_token_id=-100,
        sample_rate=16_000,
    )


def test_collate_fn(collate_fn: WhisperCollateFn):
    sample_list = [
        SpeechTextSample(speech=torch.randn(16_000), text="hello world"),
        SpeechTextSample(speech=torch.randn(8_000), text="hoge"),
    ]

    batch = collate_fn(sample_list)

    assert set(batch.keys()) == {"input_features", "decoder_input_ids", "decoder_attention_mask", "labels"}
    assert batch["input_features"].shape == (2, 128, 3000)  # (batch_size, n_mels, frame_length)
    assert batch["decoder_attention_mask"].shape == batch["decoder_input_ids"].shape
    assert batch["labels"].shape == batch["decoder_input_ids"].shape
