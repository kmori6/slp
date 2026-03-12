import pytest
import torch

from slp.modules.vits.spline_coupling_layer import SplineCouplingLayer


@pytest.fixture
def module() -> SplineCouplingLayer:
    return SplineCouplingLayer(
        input_size=2,
        hidden_size=192,
        kernel_size=3,
        num_blocks=3,
        num_bins=10,
        B=5.0,
        dropout_rate=0.0,
    )


def test_output_shape(module: SplineCouplingLayer):
    x = torch.randn(2, 2, 20)
    g = torch.randn(2, 192, 20)

    out, logdet = module(x, g=g, reverse=False)

    assert out.shape == (2, 2, 20)
    assert logdet.shape == (2,)


def test_gradient_flow(module: SplineCouplingLayer):
    x = torch.randn(2, 2, 20, requires_grad=True)
    g = torch.randn(2, 192, 20, requires_grad=True)

    out, logdet = module(x, g=g, reverse=False)
    (out.sum() + logdet.sum()).backward()

    assert x.grad is not None
    assert g.grad is not None
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} has no grad"
