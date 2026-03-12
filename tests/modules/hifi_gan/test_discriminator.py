import pytest
import torch

from slp.modules.hifi_gan.discriminator import Discriminator


@pytest.fixture
def module() -> Discriminator:
    return Discriminator()


def test_output_shape(module: Discriminator):
    x = torch.randn(2, 1, 8192)
    outs, fmaps = module(x)

    assert len(outs) == 6
    assert len(fmaps) == 6
    for o in outs:
        assert o.dim() == 2
        assert o.shape[0] == 2
    for fm in fmaps:
        assert len(fm) > 0


def test_gradient_flow(module: Discriminator):
    x = torch.randn(2, 1, 8192, requires_grad=True)

    outs, _ = module(x)
    loss = sum(out.sum() for out in outs)
    loss.backward()

    assert x.grad is not None
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} has no grad"
