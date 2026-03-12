import pytest
import torch

from slp.modules.hifi_gan.period_discriminator import PeriodDiscriminator


@pytest.fixture
def module() -> PeriodDiscriminator:
    return PeriodDiscriminator(period=3, negative_slope=0.1)


def test_output_shape(module: PeriodDiscriminator):
    x = torch.randn(2, 1, 8192)
    logits, feats = module(x)

    assert logits.dim() == 2
    assert logits.shape[0] == 2
    assert len(feats) == 6


def test_gradient_flow(module: PeriodDiscriminator):
    x = torch.randn(2, 1, 8192, requires_grad=True)

    logits, _ = module(x)
    logits.sum().backward()

    assert x.grad is not None
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} has no grad"
