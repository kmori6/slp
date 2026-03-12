import pytest
import torch

from slp.modules.transformer.feed_forward import FeedForward
from slp.modules.transformer.multi_head_attention import MultiHeadAttention
from slp.modules.transformer.post_norm_encoder_layer import PostNormEncoderLayer


@pytest.fixture
def module() -> PostNormEncoderLayer:
    mha = MultiHeadAttention(hidden_size=64, d_k=8, num_heads=8, num_kv_heads=8, dropout_rate=0.1)
    ffn = FeedForward(input_size=64, hidden_size=256, dropout_rate=0.1)
    return PostNormEncoderLayer(
        mha_module=mha,
        mha_norm_module=torch.nn.LayerNorm(64),
        ffn_module=ffn,
        ffn_norm_module=torch.nn.LayerNorm(64),
        dropout_rate=0.1,
    )


def test_output_shape(module: PostNormEncoderLayer):
    x = torch.randn(4, 10, 64)
    mask = torch.ones(4, 10, 10, dtype=torch.bool)

    output = module(x, mask)

    assert output.shape == (4, 10, 64)


def test_gradient_flow(module: PostNormEncoderLayer):
    x = torch.randn(2, 10, 64, requires_grad=True)
    mask = torch.ones(2, 10, 10, dtype=torch.bool)

    output = module(x, mask)
    output.sum().backward()

    assert x.grad is not None
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} has no grad"
