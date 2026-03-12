import pytest
import torch

from slp.modules.wavenet.residual_block import ResidualBlock


@pytest.fixture
def module() -> ResidualBlock:
    return ResidualBlock(input_size=64, kernel_size=3, dilation=1, dropout_rate=0.1)


def test_output_shape(module: ResidualBlock):
    x = torch.randn(2, 64, 100)

    residual, skip = module(x)

    assert residual.shape == (2, 64, 100)
    assert skip.shape == (2, 64, 100)


def test_gradient_flow(module: ResidualBlock):
    x = torch.randn(2, 64, 100, requires_grad=True)

    residual, skip = module(x)
    (residual + skip).sum().backward()

    assert x.grad is not None
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} has no grad"
