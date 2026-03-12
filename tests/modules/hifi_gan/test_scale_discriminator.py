import pytest
import torch

from slp.modules.hifi_gan.scale_discriminator import ScaleDiscriminator


@pytest.fixture
def module() -> ScaleDiscriminator:
    return ScaleDiscriminator(scale=1, negative_slope=0.1)


def test_output_shape(module: ScaleDiscriminator):
    x = torch.randn(2, 1, 8192)
    logits, feats = module(x)

    assert logits.dim() == 2
    assert logits.shape[0] == 2
    assert len(feats) == 7


def test_gradient_flow(module: ScaleDiscriminator):
    x = torch.randn(2, 1, 8192, requires_grad=True)

    logits, _ = module(x)
    logits.sum().backward()

    assert x.grad is not None
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} has no grad"
