import pytest
import torch

from slp.modules.vits.stochastic_duration_predictor import StochasticDurationPredictor


@pytest.fixture
def module() -> StochasticDurationPredictor:
    return StochasticDurationPredictor(
        input_size=192,
        hidden_size=192,
        kernel_size=3,
        dropout_rate=0.0,
        cond_size=64,
        num_flow_blocks=2,
        num_prior_layers=2,
        num_condition_layers=2,
        num_posterior_layers=2,
        spline_num_bins=10,
        spline_bound=5.0,
    )


def test_output_shape(module: StochasticDurationPredictor):
    x = torch.randn(2, 192, 20)
    g = torch.randn(2, 64, 20)
    d = torch.rand(2, 1, 20) + 1.0

    out = module(x, d=d, g=g, reverse=False)

    assert out.shape == (2,)


def test_gradient_flow(module: StochasticDurationPredictor):
    x = torch.randn(2, 192, 20)
    g = torch.randn(2, 64, 20)
    d = torch.rand(2, 1, 20) + 1.0
    d.requires_grad_()

    out = module(x, d=d, g=g, reverse=False)
    out.sum().backward()

    assert d.grad is not None
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} has no grad"
