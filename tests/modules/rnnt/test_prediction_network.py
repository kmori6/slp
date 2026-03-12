import pytest
import torch

from slp.modules.rnnt.prediction_network import PredictionNetwork


@pytest.fixture
def module() -> PredictionNetwork:
    return PredictionNetwork(vocab_size=128, hidden_size=64, num_layers=2, dropout_rate=0.1, blank_token_id=0)


def test_output_shape(module: PredictionNetwork):
    token = torch.randint(0, 128, (2, 10), dtype=torch.long)
    hidden_state, cell_state = module.init_state(batch_size=2, device=torch.device("cpu"))

    output, hidden_state, cell_state = module(token, hidden_state, cell_state)

    assert output.shape == (2, 10, 64)
    assert hidden_state.shape == (2, 2, 64)
    assert cell_state.shape == (2, 2, 64)


def test_gradient_flow(module: PredictionNetwork):
    token = torch.randint(0, 128, (2, 10), dtype=torch.long)
    hidden_state = torch.randn(2, 2, 64, requires_grad=True)
    cell_state = torch.randn(2, 2, 64, requires_grad=True)

    output, _, _ = module(token, hidden_state, cell_state)

    output.sum().backward()

    assert hidden_state.grad is not None
    assert cell_state.grad is not None
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} has no grad"
