import pytest
import torch

from slp.modules.conformer.block import Block


@pytest.fixture
def module() -> Block:
    return Block(input_size=64, num_heads=4, kernel_size=15, dropout_rate=0.1)


def test_output_shape(module: Block):
    x = torch.randn(2, 10, 64)
    mask = torch.ones(2, 10, 10, dtype=torch.bool)

    output = module(x, mask)

    assert output.shape == (2, 10, 64)


def test_gradient_flow(module: Block):
    x = torch.randn(2, 10, 64, requires_grad=True)
    mask = torch.ones(2, 10, 10, dtype=torch.bool)

    output = module(x, mask)

    output.sum().backward()

    assert x.grad is not None
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} has no grad"
