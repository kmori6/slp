import random
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase

from slp.dataset.speech_text_dataset import SpeechTextSample
from slp.modules.frontend.linear_spectrogram import LinearSpectrogram


@dataclass
class TextToSpeechCollateFn:
    tokenizer: PreTrainedTokenizerBase
    frontend: LinearSpectrogram
    max_length: int
    segment_size: int
    pad_token_id: int

    def __call__(self, batch: list[SpeechTextSample]) -> dict[str, torch.Tensor]:
        # text -> token ids
        texts = [sample.text for sample in batch]
        encoded = self.tokenizer(texts, truncation=True, max_length=self.max_length)
        input_ids_list = [torch.tensor(ids, dtype=torch.long) for ids in encoded["input_ids"]]
        if "attention_mask" in encoded:
            attention_mask_list = [torch.tensor(mask, dtype=torch.bool) for mask in encoded["attention_mask"]]
        else:
            attention_mask_list = [torch.ones_like(ids, dtype=torch.bool) for ids in input_ids_list]
        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.pad_token_id)
        attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)

        # speech -> linear spectrogram
        speech_list = [sample.speech if isinstance(sample, SpeechTextSample) else sample["speech"] for sample in batch]
        speech_length = torch.tensor([speech.shape[-1] for speech in speech_list], dtype=torch.long)
        speech = pad_sequence(speech_list, batch_first=True, padding_value=0.0)
        xlin, xlin_mask = self.frontend(speech, speech_length)
        xlin_length = xlin_mask.sum(dim=-1)

        # random slice start indices in latent frame units
        start_indices = torch.tensor(
            [
                random.randint(0, int(length.item()) - self.segment_size) if length > self.segment_size else 0
                for length in xlin_length
            ],
            dtype=torch.long,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "xlin": xlin,
            "xlin_mask": xlin_mask,
            "start_indices": start_indices,
            "speech": speech,
        }
