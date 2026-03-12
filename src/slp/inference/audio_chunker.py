import math
from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Chunk:
    audio: torch.Tensor
    is_last: bool


def slice_audio_with_padding(audio: torch.Tensor, start: int, end: int, total_samples: int) -> torch.Tensor:
    """Slice audio with zero-padding if the requested range is out of bounds.

    Args:
        audio (torch.Tensor): Waveform tensor of shape (1, num_samples).
        start (int): Start index of the slice.
        end (int): End index of the slice.
        total_samples (int): Total number of samples in the audio.

    Returns:
        torch.Tensor: Sliced audio with shape (1, end - start), zero-padded if out of range.
    """
    clamped_start = max(start, 0)
    clamped_end = min(end, total_samples)

    segment = audio[:, clamped_start:clamped_end]
    pad_left = clamped_start - start
    pad_right = end - clamped_end

    if pad_left > 0 or pad_right > 0:
        segment = F.pad(segment, (pad_left, pad_right))

    return segment


class AudioChunker:
    def __init__(self, chunk_ms: int, left_context_ms: int, right_context_ms: int, sample_rate: int):
        self.chunk_size = int(chunk_ms * sample_rate / 1000)
        self.left_context_size = int(left_context_ms * sample_rate / 1000)
        self.right_context_size = int(right_context_ms * sample_rate / 1000)

    def stream(self, audio: torch.Tensor) -> Iterator[Chunk]:
        """Yield fixed-size chunks with left/right context from the given waveform.

        Args:
            audio (torch.Tensor): Waveform tensor of shape (1, num_samples).

        Yields:
            Chunk: Audio chunk with zero-padded context.
        """
        total_samples = audio.shape[1]
        num_chunks = math.ceil(total_samples / self.chunk_size)

        for chunk_idx in range(num_chunks):
            center_start = chunk_idx * self.chunk_size
            center_end = min(center_start + self.chunk_size, total_samples)

            left = slice_audio_with_padding(
                audio=audio,
                start=center_start - self.left_context_size,
                end=center_start,
                total_samples=total_samples,
            )
            center = slice_audio_with_padding(
                audio=audio,
                start=center_start,
                end=center_start + self.chunk_size,
                total_samples=total_samples,
            )
            right = slice_audio_with_padding(
                audio=audio,
                start=center_end,
                end=center_end + self.right_context_size,
                total_samples=total_samples,
            )

            chunk_audio = torch.cat([left, center, right], dim=1)
            yield Chunk(audio=chunk_audio, is_last=(chunk_idx == num_chunks - 1))
