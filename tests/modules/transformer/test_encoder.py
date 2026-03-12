import pytest
import torch

from slp.modules.transformer.encoder import Encoder
from slp.modules.transformer.feed_forward import FeedForward
from slp.modules.transformer.multi_head_attention import MultiHeadAttention
from slp.modules.transformer.pre_norm_encoder_layer import PreNormEncoderLayer
from slp.modules.transformer.types import Cache


@pytest.fixture
def module() -> Encoder:
    encoder_layer = PreNormEncoderLayer(
        mha_module=MultiHeadAttention(hidden_size=64, d_k=8, num_heads=8, num_kv_heads=8, dropout_rate=0.1),
        mha_norm_module=torch.nn.LayerNorm(64),
        ffn_module=FeedForward(input_size=64, hidden_size=256, dropout_rate=0.1),
        ffn_norm_module=torch.nn.LayerNorm(64),
        dropout_rate=0.1,
    )
    return Encoder(encoder_layer=encoder_layer, num_layers=3)


def test_output_shape(module: Encoder):
    x = torch.randn(4, 10, 64)
    mask = torch.ones(4, 10, 10, dtype=torch.bool)

    output = module(x, mask)

    assert output.shape == (4, 10, 64)


def test_gradient_flow(module: Encoder):
    x = torch.randn(2, 10, 64, requires_grad=True)
    mask = torch.ones(2, 10, 10, dtype=torch.bool)

    output = module(x, mask)

    output.sum().backward()

    assert x.grad is not None
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} has no grad"


def test_predict_no_cache(module: Encoder):
    module.eval()
    x = torch.randn(1, 1, 64)
    mask = torch.ones(1, 1, 1, dtype=torch.bool)

    output, caches = module.predict(x=x, mask=mask, caches=[])

    assert output.shape == (1, 1, 64)
    assert len(caches) == module.num_layers
    for cache in caches:
        assert cache.key.shape == (1, 1, 64)
        assert cache.value.shape == (1, 1, 64)


def test_predict_cache(module: Encoder):
    module.eval()
    prev_caches = [Cache(key=torch.randn(1, 5, 64), value=torch.randn(1, 5, 64)) for _ in range(module.num_layers)]
    x = torch.randn(1, 1, 64)
    mask = torch.ones(1, 1, 6, dtype=torch.bool)

    output, caches = module.predict(x=x, mask=mask, caches=prev_caches)

    assert output.shape == (1, 1, 64)
    assert len(caches) == module.num_layers
    for cache in caches:
        assert cache.key.shape == (1, 6, 64)
        assert cache.value.shape == (1, 6, 64)
