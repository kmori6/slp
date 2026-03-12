import torch

from slp.modules.transformer.positional_encoding import PositionalEncoding, sinusoidal_positional_encoding


def test_sinusoidal_positional_encoding():
    d_model = 8
    max_length = 6
    base = 10000.0
    positions = torch.arange(max_length, dtype=torch.float32)[:, None]
    frequencies = base ** (torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
    angles = positions / frequencies
    expected = torch.zeros(max_length, d_model, dtype=torch.float32)
    expected[:, 0::2] = torch.sin(angles)
    expected[:, 1::2] = torch.cos(angles)

    pe = sinusoidal_positional_encoding(d_model=d_model, max_length=max_length, base=base)

    assert torch.allclose(pe, expected)


def test_output_shape():
    positional_encoding = PositionalEncoding(hidden_size=16, max_length=32)
    x = torch.randn(4, 20, 16)

    output = positional_encoding(x)

    assert output.shape == (1, 20, 16)


def test_extend_output_shape():
    positional_encoding = PositionalEncoding(hidden_size=16, max_length=8)
    x = torch.randn(4, 12, 16)

    output = positional_encoding(x)

    assert output.shape == (1, 12, 16)
    assert positional_encoding.max_length == 12
