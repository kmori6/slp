import pytest
import torch

from slp.modules.vits.dds_conv import DDSConv


@pytest.fixture
def module() -> DDSConv:
    return DDSConv(
        hidden_size=192,
        kernel_size=3,
        dilation=1,
        dropout_rate=0.1,
    )


def test_output_shape(module: DDSConv):
    x = torch.randn(2, 192, 20)

    out = module(x)

    assert out.shape == (2, 192, 20)


def test_gradient_flow(module: DDSConv):
    x = torch.randn(2, 192, 20, requires_grad=True)

    out = module(x)
    out.sum().backward()

    assert x.grad is not None
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} has no grad"
