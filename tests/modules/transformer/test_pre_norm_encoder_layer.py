import pytest
import torch

from slp.modules.transformer.feed_forward import FeedForward
from slp.modules.transformer.multi_head_attention import MultiHeadAttention
from slp.modules.transformer.pre_norm_encoder_layer import PreNormEncoderLayer
from slp.modules.transformer.types import Cache


@pytest.fixture
def module() -> PreNormEncoderLayer:
    mha = MultiHeadAttention(hidden_size=64, d_k=8, num_heads=8, num_kv_heads=8, dropout_rate=0.1)
    ffn = FeedForward(input_size=64, hidden_size=256, dropout_rate=0.1)
    return PreNormEncoderLayer(
        mha_module=mha,
        mha_norm_module=torch.nn.LayerNorm(64),
        ffn_module=ffn,
        ffn_norm_module=torch.nn.LayerNorm(64),
        dropout_rate=0.1,
    )


def test_output_shape(module: PreNormEncoderLayer):
    x = torch.randn(4, 10, 64)
    mask = torch.ones(4, 10, 10, dtype=torch.bool)

    output = module(x, mask)

    assert output.shape == (4, 10, 64)


def test_gradient_flow(module: PreNormEncoderLayer):
    x = torch.randn(2, 10, 64, requires_grad=True)
    mask = torch.ones(2, 10, 10, dtype=torch.bool)

    output = module(x, mask)

    output.sum().backward()

    assert x.grad is not None
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} has no grad"


def test_predict_no_cache(module: PreNormEncoderLayer):
    module.eval()
    x = torch.randn(1, 1, 64)
    mask = torch.ones(1, 1, 1, dtype=torch.bool)

    output, cache = module.predict(x=x, mask=mask, cache=None)

    assert output.shape == (1, 1, 64)
    assert cache.key.shape == (1, 1, 64)
    assert cache.value.shape == (1, 1, 64)


def test_predict_cache(module: PreNormEncoderLayer):
    module.eval()
    prev_cache = Cache(key=torch.randn(1, 5, 64), value=torch.randn(1, 5, 64))
    x = torch.randn(1, 1, 64)
    mask = torch.ones(1, 1, 6, dtype=torch.bool)

    output, cache = module.predict(x=x, mask=mask, cache=prev_cache)

    assert output.shape == (1, 1, 64)
    assert cache.key.shape == (1, 6, 64)
    assert cache.value.shape == (1, 6, 64)
