import pytest
import torch

from slp.modules.vits.flow import Flow


@pytest.fixture
def module() -> Flow:
    return Flow(
        input_size=192,
        hidden_size=192,
        kernel_size=5,
        dilation_rate=1,
        num_blocks=4,
        num_layers=4,
        dropout_rate=0.0,
        cond_size=0,
    )


def test_output_shape(module: Flow):
    x = torch.randn(2, 192, 10)

    out = module(x)

    assert out.shape == (2, 192, 10)


def test_gradient_flow(module: Flow):
    x = torch.randn(2, 192, 10, requires_grad=True)

    out = module(x)
    out.sum().backward()

    assert x.grad is not None
    excluded_prefixes = [
        f"coupling_layers.{layer_idx}.blocks.{len(layer.blocks) - 1}.res_conv"
        for layer_idx, layer in enumerate(module.coupling_layers)
    ]
    for name, parameter in module.named_parameters():
        if any(name.startswith(prefix) for prefix in excluded_prefixes):
            continue
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} has no grad"


def test_reverse(module: Flow):
    x = torch.randn(2, 192, 10)

    out = module(x)
    out_reversed = module(out, reverse=True)

    assert torch.allclose(x, out_reversed)
