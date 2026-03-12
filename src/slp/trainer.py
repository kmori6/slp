from dataclasses import dataclass
from logging import getLogger
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = getLogger(__name__)


@dataclass
class TrainerArguments:
    device: torch.device
    out_dir: Path
    epochs: int
    checkpoint_path: Path | None = None
    grad_accum_steps: int = 1
    max_norm: float = 1.0
    log_steps: int = 10


class Trainer:
    """Trainer for training a model.

    Tuning guide: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html

    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        scaler: GradScaler,
        args: TrainerArguments,
    ):
        self.args = args
        self.model = model.to(self.args.device)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.start_epoch = 0
        self.best_valid_loss = float("inf")

    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        for step, batch in tqdm(
            enumerate(self.train_dataloader, 1), total=len(self.train_dataloader), dynamic_ncols=True
        ):
            # amp reference: https://docs.pytorch.org/docs/stable/notes/amp_examples.html
            with autocast("cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16):
                outputs = self.model(**{k: v.to(self.args.device) for k, v in batch.items()})
                if "loss" not in outputs:
                    raise ValueError("Model must return a loss")
                loss = outputs["loss"] / self.args.grad_accum_steps
            self.scaler.scale(loss).backward()
            if step % self.args.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
            if step % self.args.log_steps == 0:
                msg = f"lr: {self.scheduler.get_last_lr()[0]:.6f}, "
                for k, v in outputs.items():
                    msg += f"{k}: {v.item():.3f}, "
                tqdm.write(msg)
            epoch_loss += outputs["loss"].item()
        return epoch_loss / len(self.train_dataloader)

    @torch.inference_mode()
    def validate_epoch(self):
        self.model.eval()
        epoch_loss = 0.0
        for batch in tqdm(
            self.valid_dataloader, desc="validating", total=len(self.valid_dataloader), leave=False, dynamic_ncols=True
        ):
            outputs = self.model(**{k: v.to(self.args.device) for k, v in batch.items()})
            if "loss" not in outputs:
                raise ValueError("Model must return a loss")
            epoch_loss += outputs["loss"].item()
        return epoch_loss / len(self.valid_dataloader)

    def save_checkpoint(self, epoch: int):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_valid_loss": self.best_valid_loss,
        }
        torch.save(state, self.args.out_dir / f"checkpoint_epoch_{epoch}.pt")
        # Remove old checkpoints, keep only the latest
        for old_ckpt in self.args.out_dir.glob("checkpoint_epoch_*.pt"):
            if old_ckpt.name != f"checkpoint_epoch_{epoch}.pt":
                old_ckpt.unlink()

    def load_checkpoint(self):
        if self.args.checkpoint_path is None:
            raise ValueError("args.checkpoint_path must be set to load a checkpoint")
        checkpoint = torch.load(self.args.checkpoint_path, map_location=self.args.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.best_valid_loss = checkpoint["best_valid_loss"]
        self.start_epoch = checkpoint["epoch"]

    def train(self):
        self.args.out_dir.mkdir(exist_ok=True)
        for epoch in tqdm(range(self.start_epoch, self.args.epochs), desc="Training", dynamic_ncols=True):
            train_loss = self.train_epoch()
            valid_loss = self.validate_epoch()
            tqdm.write(
                f"epoch {epoch + 1}/{self.args.epochs}, train loss: {train_loss:.3f}, valid loss: {valid_loss:.3f}"
            )
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), self.args.out_dir / "best_model.pt")
            self.save_checkpoint(epoch + 1)
