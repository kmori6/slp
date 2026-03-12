import pytest
import torch

from slp.modules.hifi_gan.res_block import ResBlock


@pytest.fixture
def module() -> ResBlock:
    return ResBlock(hidden_size=64, kernel_size=3, dilations=[[1, 1], [3, 1], [5, 1]], negative_slope=0.1)


def test_output_shape(module: ResBlock):
    x = torch.randn(2, 64, 100)

    output = module(x)

    assert output.shape == x.shape


def test_gradient_flow(module: ResBlock):
    x = torch.randn(2, 64, 100, requires_grad=True)

    output = module(x)
    output.sum().backward()

    assert x.grad is not None
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} has no grad"
