from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class EpochDecayLRScheduler(LRScheduler):
    def __init__(self, optimizer: Optimizer, gamma: float, last_epoch: int = -1):
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        return [base_lr * (self.gamma**self.last_epoch) for base_lr in self.base_lrs]
