import pytest
import torch

from slp.modules.vits.affine_coupling_layer import AffineCouplingLayer


@pytest.fixture
def module() -> AffineCouplingLayer:
    return AffineCouplingLayer(
        input_size=192,
        hidden_size=192,
        kernel_size=5,
        dilation_rate=1,
        num_blocks=4,
        dropout_rate=0.0,
        cond_size=0,
    )


def test_output_shape(module: AffineCouplingLayer):
    x = torch.randn(2, 192, 10)
    out = module(x)
    assert out.shape == (2, 192, 10)


def test_gradient_flow(module: AffineCouplingLayer):
    x = torch.randn(2, 192, 10, requires_grad=True)

    out = module(x)
    out.sum().backward()

    assert x.grad is not None
    last_res_conv_prefix = f"blocks.{len(module.blocks) - 1}.res_conv"
    for name, parameter in module.named_parameters():
        if name.startswith(last_res_conv_prefix):
            continue
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} has no grad"
