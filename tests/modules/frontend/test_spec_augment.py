import pytest
import torch

from slp.modules.frontend.spec_augment import SpecAugment


@pytest.fixture
def module() -> SpecAugment:
    return SpecAugment(num_freq_masks=2, num_time_masks=2, max_freq_mask_size=10, max_time_mask_size=20)


def test_output_shape(module: SpecAugment):
    module.train()
    x = torch.randn(4, 100, 80)

    output = module(x)

    assert output.shape == x.shape


def test_no_augmentation(module: SpecAugment):
    module.eval()
    x = torch.randn(4, 100, 80)

    output = module(x)

    assert torch.equal(output, x)
