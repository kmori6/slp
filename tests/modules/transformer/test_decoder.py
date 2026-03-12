import pytest
import torch

from slp.modules.transformer.decoder import Decoder
from slp.modules.transformer.feed_forward import FeedForward
from slp.modules.transformer.multi_head_attention import MultiHeadAttention
from slp.modules.transformer.pre_norm_decoder_layer import PreNormDecoderLayer
from slp.modules.transformer.types import Cache


@pytest.fixture
def module() -> Decoder:
    decoder_layer = PreNormDecoderLayer(
        masked_mha_module=MultiHeadAttention(hidden_size=64, d_k=8, num_heads=8, num_kv_heads=8, dropout_rate=0.1),
        masked_mha_norm_module=torch.nn.LayerNorm(64),
        mha_module=MultiHeadAttention(hidden_size=64, d_k=8, num_heads=8, num_kv_heads=8, dropout_rate=0.1),
        mha_norm_module=torch.nn.LayerNorm(64),
        ffn_module=FeedForward(input_size=64, hidden_size=256, dropout_rate=0.1),
        ffn_norm_module=torch.nn.LayerNorm(64),
        dropout_rate=0.1,
    )
    return Decoder(decoder_layer=decoder_layer, num_layers=3)


def test_output_shape(module: Decoder):
    x_enc = torch.randn(4, 12, 64)
    x_dec = torch.randn(4, 8, 64)
    mask_enc = torch.ones(4, 8, 12, dtype=torch.bool)
    mask_dec = torch.ones(4, 8, 8, dtype=torch.bool)

    output = module(x_enc, x_dec, mask_enc, mask_dec)

    assert output.shape == (4, 8, 64)


def test_gradient_flow(module: Decoder):
    x_enc = torch.randn(2, 10, 64, requires_grad=True)
    x_dec = torch.randn(2, 8, 64, requires_grad=True)
    mask_enc = torch.ones(2, 8, 10, dtype=torch.bool)
    mask_dec = torch.ones(2, 8, 8, dtype=torch.bool)

    output = module(x_enc, x_dec, mask_enc, mask_dec)

    output.sum().backward()

    assert x_enc.grad is not None
    assert x_dec.grad is not None
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} has no grad"


def test_predict_no_cache(module: Decoder):
    module.eval()
    x_enc = torch.randn(1, 12, 64)
    x_dec = torch.randn(1, 1, 64)

    output, caches = module.predict(x_enc=x_enc, x_dec=x_dec, caches=[])

    assert output.shape == (1, 1, 64)
    assert len(caches) == module.num_layers
    for cache in caches:
        assert cache.key.shape == (1, 1, 64)
        assert cache.value.shape == (1, 1, 64)


def test_predict_cache(module: Decoder):
    module.eval()
    prev_caches = [Cache(key=torch.randn(1, 5, 64), value=torch.randn(1, 5, 64)) for _ in range(module.num_layers)]
    x_enc = torch.randn(1, 12, 64)
    x_dec = torch.randn(1, 1, 64)

    output, caches = module.predict(x_enc=x_enc, x_dec=x_dec, caches=prev_caches)

    assert output.shape == (1, 1, 64)
    assert len(caches) == module.num_layers
    for cache in caches:
        assert cache.key.shape == (1, 6, 64)
        assert cache.value.shape == (1, 6, 64)
