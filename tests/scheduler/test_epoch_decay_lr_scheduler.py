import pytest
import torch
from torch.optim import Adam

from slp.scheduler.epoch_decay_lr_scheduler import EpochDecayLRScheduler


@pytest.fixture
def scheduler():
    model = torch.nn.Linear(10, 10)
    optimizer = Adam(model.parameters(), lr=1.0)
    return EpochDecayLRScheduler(optimizer, gamma=0.1)


def test_lr_schedule(scheduler: EpochDecayLRScheduler):

    # warmup start
    assert scheduler.get_lr() == [1.0]

    # decay: epoch 5
    for _ in range(5):
        scheduler.step()
    assert scheduler.get_lr() == pytest.approx([1e-5])

    # decay: step 100
    for _ in range(95):
        scheduler.step()
    assert scheduler.get_lr() == pytest.approx([1e-100])
