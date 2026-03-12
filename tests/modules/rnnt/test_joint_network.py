import pytest
import torch

from slp.modules.rnnt.joint_network import JointNetwork


@pytest.fixture
def module() -> JointNetwork:
    return JointNetwork(vocab_size=128, encoder_size=64, predictor_size=32, hidden_size=96, dropout_rate=0.1)


def test_output_shape(module: JointNetwork):
    x_enc = torch.randn(2, 10, 1, 64)
    x_prd = torch.randn(2, 1, 12, 32)

    output = module(x_enc, x_prd)

    assert output.shape == (2, 10, 12, 128)


def test_gradient_flow(module: JointNetwork):
    x_enc = torch.randn(2, 10, 1, 64, requires_grad=True)
    x_prd = torch.randn(2, 1, 12, 32, requires_grad=True)

    output = module(x_enc, x_prd)

    output.sum().backward()

    assert x_enc.grad is not None
    assert x_prd.grad is not None
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} has no grad"
