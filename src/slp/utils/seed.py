import random

import numpy as np
import torch


def fix_seed(seed: int = 0, use_deterministic_algorithms: bool = True):
    """Fix the random seed for reproducibility.

    https://pytorch.org/docs/stable/notes/randomness.html

    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(use_deterministic_algorithms)
