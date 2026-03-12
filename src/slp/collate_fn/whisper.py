from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase, WhisperFeatureExtractor

from slp.dataset.speech_text_dataset import SpeechTextSample


@dataclass
class WhisperCollateFn:
    tokenizer: PreTrainedTokenizerBase
    frontend: WhisperFeatureExtractor
    max_length: int
    pad_token_id: int
    ignore_token_id: int = -100
    sample_rate: int = 16_000

    def __call__(self, sample_list: list[SpeechTextSample]) -> dict[str, torch.Tensor]:
        input_features_list = []
        decoder_input_ids_list = []
        decoder_attention_mask_list = []
        labels_list = []

        for sample in sample_list:
            speech = sample.speech if isinstance(sample, SpeechTextSample) else sample["speech"]
            text = sample.text if isinstance(sample, SpeechTextSample) else sample["text"]

            # speech -> log mel spectrogram: WhisperFeatureExtractor automatically pads/truncates to 3000 frames
            processed = self.frontend(speech, sampling_rate=self.sample_rate, return_tensors="pt")
            input_features = processed.input_features.squeeze(0)  # (batch_size, n_mels, frames) -> (n_mels, frames)
            input_features_list.append(input_features)

            # text -> token ids
            tokenized = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True)
            token_ids = tokenized.input_ids.squeeze(0)  # (seq_len,)
            decoder_attention_mask = tokenized.attention_mask.squeeze(0)  # (seq_len,)

            # Shift decoder input ids and labels for causal modeling.
            decoder_input_ids = token_ids[:-1]
            decoder_attention_mask = decoder_attention_mask[:-1]
            labels = token_ids[1:]

            decoder_input_ids_list.append(decoder_input_ids)
            decoder_attention_mask_list.append(decoder_attention_mask)
            labels_list.append(labels)

        decoder_input_ids = pad_sequence(decoder_input_ids_list, batch_first=True, padding_value=self.pad_token_id)
        decoder_attention_mask = pad_sequence(decoder_attention_mask_list, batch_first=True, padding_value=0)
        labels = pad_sequence(labels_list, batch_first=True, padding_value=self.ignore_token_id)

        batch = {
            "input_features": torch.stack(input_features_list),
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
        }

        return batch
