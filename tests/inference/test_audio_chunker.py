import torch

from slp.inference.audio_chunker import AudioChunker


def test_stream_outputs_expected_chunks():
    chunker = AudioChunker(chunk_ms=4, left_context_ms=2, right_context_ms=3, sample_rate=1000)
    audio = torch.arange(10, dtype=torch.float32).unsqueeze(0)

    chunks = list(chunker.stream(audio))

    assert len(chunks) == 3
    assert [chunk.is_last for chunk in chunks] == [False, False, True]
    assert torch.equal(chunks[0].audio, torch.tensor([[0, 0, 0, 1, 2, 3, 4, 5, 6]], dtype=torch.float32))
    assert torch.equal(chunks[1].audio, torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9, 0]], dtype=torch.float32))
    assert torch.equal(chunks[2].audio, torch.tensor([[6, 7, 8, 9, 0, 0, 0, 0, 0]], dtype=torch.float32))


def test_stream_returns_no_chunks_for_empty_audio():
    chunker = AudioChunker(chunk_ms=4, left_context_ms=2, right_context_ms=3, sample_rate=1000)
    audio = torch.empty(1, 0)

    chunks = list(chunker.stream(audio))

    assert len(chunks) == 0
