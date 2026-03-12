import pytest
import torch

from slp.modules.vits.condition_encoder import ConditionEncoder


@pytest.fixture
def module() -> ConditionEncoder:
    return ConditionEncoder(
        input_size=192,
        hidden_size=192,
        kernel_size=3,
        num_layers=3,
        dropout_rate=0.1,
        cond_size=64,
    )


def test_output_shape(module: ConditionEncoder):
    x = torch.randn(2, 192, 20)
    g = torch.randn(2, 64, 20)

    out = module(x, g=g)

    assert out.shape == (2, 192, 20)


def test_gradient_flow(module: ConditionEncoder):
    x = torch.randn(2, 192, 20, requires_grad=True)
    g = torch.randn(2, 64, 20, requires_grad=True)

    out = module(x, g=g)
    out.sum().backward()

    assert x.grad is not None
    assert g.grad is not None
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} has no grad"
