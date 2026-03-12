import pytest
import torch

from slp.modules.conformer.convolution_subsampling import ConvolutionSubsampling


@pytest.fixture
def module() -> ConvolutionSubsampling:
    return ConvolutionSubsampling(output_size=64)


def test_output_shape(module: ConvolutionSubsampling):
    batch_size, frame_length, input_size = 2, 100, 80
    x = torch.randn(batch_size, frame_length, input_size)
    lengths = torch.full((batch_size,), frame_length, dtype=torch.long)

    output, output_mask = module(x, lengths)

    expected_frames = ((frame_length - 1) // 2 - 1) // 2
    expected_features = 64 * (((input_size - 1) // 2 - 1) // 2)
    assert output.shape == (batch_size, expected_frames, expected_features)
    assert output_mask.shape == (batch_size, expected_frames)


def test_gradient_flow(module: ConvolutionSubsampling):
    x = torch.randn(2, 100, 80, requires_grad=True)
    lengths = torch.full((2,), 100, dtype=torch.long)

    output, _ = module(x, lengths)

    output.sum().backward()

    assert x.grad is not None
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} has no grad"
