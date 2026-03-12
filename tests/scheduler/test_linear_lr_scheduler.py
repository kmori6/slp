import pytest
import torch
from torch.optim import Adam

from slp.scheduler.linear_lr_scheduler import LinearLRScheduler


@pytest.fixture
def scheduler():
    model = torch.nn.Linear(10, 10)
    optimizer = Adam(model.parameters(), lr=1.0)
    return LinearLRScheduler(optimizer, total_steps=100, warmup_steps=10)


def test_lr_schedule(scheduler: LinearLRScheduler):

    # warmup start
    assert scheduler.get_lr() == [0.0]

    # warmup middle: step 5
    for _ in range(5):
        scheduler.step()
    assert scheduler.get_lr() == [0.5]

    # warmup end: step 10
    for _ in range(5):
        scheduler.step()
    assert scheduler.get_lr() == [1.0]

    # decay middle: step 50
    for _ in range(40):
        scheduler.step()
    assert scheduler.get_lr()[0] == 50 / 90

    # decay end: step 100
    for _ in range(50):
        scheduler.step()
    assert scheduler.get_lr() == [0.0]
