import pytest
import torch

from slp.modules.vits.text_encoder import TextEncoder


@pytest.fixture
def module() -> TextEncoder:
    return TextEncoder(
        vocab_size=50,
        hidden_size=64,
        ffn_size=256,
        num_heads=2,
        num_layers=2,
        dropout_rate=0.1,
        window_size=4,
    )


def test_output_shape(module: TextEncoder):
    x = torch.randint(0, 50, (2, 10))
    mask = torch.ones(2, 10, dtype=torch.bool)

    output = module(x, mask)

    assert output.shape == (2, 10, 64)


def test_gradient_flow(module: TextEncoder):
    x = torch.randint(0, 50, (2, 10))
    mask = torch.ones(2, 10, dtype=torch.bool)

    output = module(x, mask)
    output.sum().backward()

    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} has no grad"
