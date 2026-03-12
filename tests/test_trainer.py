import pytest
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

from slp.trainer import Trainer, TrainerArguments


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        out = self.linear(x)
        loss = out.sum()
        return {"loss": loss, "output": out}


@pytest.fixture
def trainer(tmp_path):
    model = DummyModel()
    dataset = TensorDataset(torch.randn(10, 10))
    train_loader = DataLoader(dataset, batch_size=4, collate_fn=lambda b: {"x": torch.stack([x[0] for x in b])})
    valid_loader = DataLoader(dataset, batch_size=4, collate_fn=lambda b: {"x": torch.stack([x[0] for x in b])})
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    scaler = GradScaler(enabled=False)
    args = TrainerArguments(
        device=torch.device("cpu"),
        out_dir=tmp_path,
        epochs=2,
        grad_accum_steps=1,
        log_steps=100,
    )
    return Trainer(
        model=model,
        train_dataloader=train_loader,
        valid_dataloader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        args=args,
    )


def test_train_epoch(trainer: Trainer):

    loss = trainer.train_epoch()

    assert isinstance(loss, float)


def test_validate_epoch(trainer: Trainer):

    loss = trainer.validate_epoch()

    assert isinstance(loss, float)


def test_save_checkpoint(trainer: Trainer, tmp_path):

    trainer.save_checkpoint(1)

    assert (tmp_path / "checkpoint_epoch_1.pt").exists()


def test_load_checkpoint(trainer: Trainer, tmp_path):
    trainer.save_checkpoint(1)
    trainer.args.checkpoint_path = tmp_path / "checkpoint_epoch_1.pt"

    trainer.load_checkpoint()

    assert trainer.start_epoch == 1


def test_train(trainer: Trainer, tmp_path):

    trainer.train()

    assert (tmp_path / "best_model.pt").exists()
    assert (tmp_path / f"checkpoint_epoch_{trainer.args.epochs}.pt").exists()
