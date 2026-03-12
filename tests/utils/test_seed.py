import random

import numpy as np
import torch

from slp.utils.seed import fix_seed


def test_fix_seed() -> None:
    fix_seed(1234)
    py_1 = random.random()
    np_1 = np.random.rand(4)
    torch_1 = torch.randn(4)

    fix_seed(1234)
    py_2 = random.random()
    np_2 = np.random.rand(4)
    torch_2 = torch.randn(4)

    assert py_1 == py_2
    assert np.allclose(np_1, np_2)
    assert torch.allclose(torch_1, torch_2)
