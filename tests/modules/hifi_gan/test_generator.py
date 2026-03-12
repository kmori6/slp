import pytest
import torch

from slp.modules.hifi_gan.generator import Generator


@pytest.fixture
def module() -> Generator:
    return Generator(input_size=80)


def test_output_shape(module: Generator):
    x = torch.randn(2, 80, 10)
    output = module(x)

    assert output.shape == (2, 1, 10 * 256)


def test_gradient_flow(module: Generator):
    x = torch.randn(2, 80, 10, requires_grad=True)

    output = module(x)
    output.sum().backward()

    assert x.grad is not None
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} has no grad"
