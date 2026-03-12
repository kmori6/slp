from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
from datasets import load_dataset
from torch.utils.data import Dataset


@dataclass
class SpeechTextSample:
    speech: torch.Tensor
    text: str


class SpeechTextDataset(Dataset[SpeechTextSample]):
    def __init__(self, data_file: str, sample_rate: int = 16_000) -> None:
        if Path(data_file).suffix.lower() != ".json":
            raise ValueError(f"Data file must be a JSON file: {data_file}")
        self.sample_rate = sample_rate
        self.dataset = load_dataset("json", data_files=data_file, split="train")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> SpeechTextSample:
        sample = self.dataset[idx]
        speech, sample_rate = torchaudio.load_with_torchcodec(sample["audio_path"])
        if sample_rate != self.sample_rate:
            speech = torchaudio.functional.resample(speech, orig_freq=sample_rate, new_freq=self.sample_rate)
        speech = speech.squeeze()  # (1, sample_length) -> (sample_length,)
        return SpeechTextSample(speech, sample["text"])
