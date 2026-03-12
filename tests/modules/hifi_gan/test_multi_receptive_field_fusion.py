import pytest
import torch

from slp.modules.hifi_gan.multi_receptive_field_fution import MultiReceptiveFieldFusion


@pytest.fixture
def module() -> MultiReceptiveFieldFusion:
    return MultiReceptiveFieldFusion(
        hidden_dim=64,
        negative_slope=0.1,
        kernel_sizes=[3, 7, 11],
        dilations_list=[
            [[1, 1], [3, 1], [5, 1]],
            [[1, 1], [3, 1], [5, 1]],
            [[1, 1], [3, 1], [5, 1]],
        ],
    )


def test_output_shape(module: MultiReceptiveFieldFusion):
    x = torch.randn(2, 64, 50)

    output = module(x)

    assert output.shape == x.shape


def test_gradient_flow(module: MultiReceptiveFieldFusion):
    x = torch.randn(2, 64, 50, requires_grad=True)

    output = module(x)
    output.sum().backward()

    assert x.grad is not None
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, f"{name} has no grad"
