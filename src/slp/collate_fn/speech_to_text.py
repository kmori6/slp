from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import TokenizersBackend

from slp.dataset.speech_text_dataset import SpeechTextSample
from slp.modules.frontend.log_mel_spectrogram import LogMelSpectrogram
from slp.modules.frontend.spec_augment import SpecAugment


@dataclass
class SpeechToTextCollateFn:
    tokenizer: TokenizersBackend
    frontend: LogMelSpectrogram
    max_length: int
    ignore_token_id: int = -100
    spec_augment: SpecAugment | None = None

    def __call__(self, batch: list[SpeechTextSample]) -> dict[str, torch.Tensor]:
        """Collate function for speech-to-text training.

        Args:
            batch (list[SpeechTextSample]): List of samples, each with "speech" (Tensor) and "text" (str).

        Returns:
            dict[str, torch.Tensor]: Dictionary containing:
                - input_values: Log mel-spectrogram (batch_size, frame_length, mel_size).
                - attention_mask: Frame-level mask (batch_size, frame_length).
                - labels: Token ids padded with ignore_token_id (batch_size, max_label_length).
        """
        # speech -> log mel-spectrogram
        speech_list = [sample.speech for sample in batch]
        lengths = torch.tensor([len(speech) for speech in speech_list])
        speech = pad_sequence(speech_list, batch_first=True)
        input_values, attention_mask = self.frontend(speech, lengths)

        # spec augment
        if self.spec_augment is not None:
            input_values = self.spec_augment(input_values)

        # text -> token ids labels
        text_list = [sample.text for sample in batch]
        encoded = self.tokenizer(text_list, truncation=True, max_length=self.max_length)
        token_ids = [torch.tensor(ids, dtype=torch.long) for ids in encoded["input_ids"]]
        labels = pad_sequence(token_ids, batch_first=True, padding_value=self.ignore_token_id)

        return {"input_values": input_values, "attention_mask": attention_mask, "labels": labels}
