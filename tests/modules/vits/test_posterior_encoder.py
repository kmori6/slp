import pytest
import torch

from slp.modules.vits.posterior_encoder import PosteriorEncoder


@pytest.fixture
def module() -> PosteriorEncoder:
    return PosteriorEncoder(
        input_size=80,
        output_size=192,
        hidden_size=192,
        kernel_size=5,
        dilation_rate=1,
        num_layers=4,
        dropout_rate=0.1,
        cond_size=0,
    )


def test_output_shape(module: PosteriorEncoder):
    x = torch.randn(2, 80, 50)

    z, mean, log_std = module(x)

    assert z.shape == (2, 192, 50)
    assert mean.shape == (2, 192, 50)
    assert log_std.shape == (2, 192, 50)


def test_gradient_flow(module: PosteriorEncoder):
    x = torch.randn(2, 80, 50, requires_grad=True)

    z, mean, log_std = module(x)
    (z + mean + log_std).sum().backward()

    assert x.grad is not None
    last_res_conv_prefix = f"blocks.{len(module.blocks) - 1}.res_conv"
    for name, parameter in module.named_parameters():
        if name.startswith(last_res_conv_prefix):
            continue
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} has no grad"
