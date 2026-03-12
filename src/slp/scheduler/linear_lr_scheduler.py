from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class LinearLRScheduler(LRScheduler):
    def __init__(self, optimizer: Optimizer, total_steps: int, warmup_steps: int, last_epoch: int = -1):
        # LRShceduler calls get_lr() in __init__, so we need to check the arguments before calling super().__init__()
        if warmup_steps <= 0:
            raise ValueError(f"warmup_steps must be > 0, got {warmup_steps}")
        if total_steps <= warmup_steps:
            raise ValueError(f"total_steps must be > warmup_steps, got {total_steps} <= {warmup_steps}")
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Compute learning rate using linear warmup and linear decay.

        lr = base_lr * step / warmup_steps, if step < warmup_steps
             base_lr * (total_steps - step) / (total_steps - warmup_steps) else

        Returns:
            list[float]: List of learning rates for each parameter group.
        """
        step = self.last_epoch
        if step < self.warmup_steps:
            scale = step / self.warmup_steps
        else:
            scale = (self.total_steps - step) / (self.total_steps - self.warmup_steps)
        return [base_lr * scale for base_lr in self.base_lrs]
