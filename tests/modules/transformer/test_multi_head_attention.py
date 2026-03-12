import pytest
import torch

from slp.modules.transformer.multi_head_attention import MultiHeadAttention


@pytest.fixture
def module() -> MultiHeadAttention:
    return MultiHeadAttention(hidden_size=64, d_k=8, num_heads=8, num_kv_heads=8, dropout_rate=0.1)


def test_output_shape(module: MultiHeadAttention):
    x = torch.randn(4, 10, 64)
    mask = torch.ones(4, 10, 10, dtype=torch.bool)

    output = module(x, x, x, mask)

    assert output.shape == (4, 10, 64)


def test_gradient_flow(module: MultiHeadAttention):
    q = torch.randn(2, 10, 64, requires_grad=True)
    k = torch.randn(2, 10, 64, requires_grad=True)
    v = torch.randn(2, 10, 64, requires_grad=True)
    mask = torch.ones(2, 10, 10, dtype=torch.bool)

    output = module(q, k, v, mask)
    loss = output.sum()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} has no grad"
