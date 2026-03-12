import pytest
import torch

from slp.modules.conformer.streaming_encoder import StreamingEncoder


@pytest.fixture
def module() -> StreamingEncoder:
    return StreamingEncoder(
        input_size=80,
        hidden_size=64,
        num_heads=4,
        kernel_size=15,
        num_blocks=1,
        dropout_rate=0.1,
        min_chunk_size=2,
        max_chunk_size=8,
    )


def test_output_shape(module: StreamingEncoder):
    batch_size, frame_length = 2, 100
    x = torch.randn(batch_size, frame_length, 80)
    lengths = torch.full((batch_size,), frame_length, dtype=torch.long)

    output, output_mask = module(x, lengths)

    expected_frames = ((frame_length - 1) // 2 - 1) // 2
    assert output.shape == (batch_size, expected_frames, 64)
    assert output_mask.shape == (batch_size, expected_frames)


def test_gradient_flow(module: StreamingEncoder):
    x = torch.randn(2, 100, 80, requires_grad=True)
    lengths = torch.full((2,), 100, dtype=torch.long)

    output, _ = module(x, lengths)

    output.sum().backward()

    assert x.grad is not None
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} has no grad"
