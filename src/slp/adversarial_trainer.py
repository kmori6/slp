import json
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

from slp.utils.stats import average_stats, tensor_to_float_stats

logger = getLogger(__name__)


@dataclass
class AdversarialTrainerArguments:
    device: torch.device
    out_dir: Path
    epochs: int
    checkpoint_path: Path | None = None
    grad_accum_steps: int = 1
    max_norm: float = 1.0
    log_steps: int = 10


class AdversarialTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        generator_optimizer: Optimizer,
        discriminator_optimizer: Optimizer,
        generator_scheduler: LRScheduler,
        discriminator_scheduler: LRScheduler,
        scaler: GradScaler,
        args: AdversarialTrainerArguments,
    ):
        if not hasattr(model, "generator") or not isinstance(model.generator, nn.Module):
            raise TypeError("model must have nn.Module generator")
        if not hasattr(model, "discriminator") or not isinstance(model.discriminator, nn.Module):
            raise TypeError("model must have nn.Module discriminator")

        self.args = args
        self.model = model.to(self.args.device)
        self.generator = model.generator
        self.discriminator = model.discriminator
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_scheduler = generator_scheduler
        self.discriminator_scheduler = discriminator_scheduler
        self.scaler = scaler
        self.autocast_device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.start_epoch = 0
        self.best_valid_loss = float("inf")
        self.stats: dict[int, dict[str, dict[str, float]]] = {}

    def train_epoch(self) -> tuple[dict[str, float], dict[str, float]]:
        g_stats_list = []
        d_stats_list = []
        self.model.train()
        self.generator_optimizer.zero_grad(set_to_none=True)
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        for step, batch in tqdm(
            enumerate(self.train_dataloader, 1), total=len(self.train_dataloader), leave=False, dynamic_ncols=True
        ):
            batch = {k: v.to(self.args.device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # generator step
            for param in self.discriminator.parameters():
                param.requires_grad_(False)
            for param in self.generator.parameters():
                param.requires_grad_(True)
            with autocast(self.autocast_device_type, dtype=torch.bfloat16):
                g_stats = self.model(**batch, phase="generator")
                if "loss" not in g_stats:
                    raise KeyError("model output dict must contain 'loss' key for generator phase")
                g_loss = g_stats["loss"] / self.args.grad_accum_steps
            self.scaler.scale(g_loss).backward()

            # discriminator step
            for param in self.discriminator.parameters():
                param.requires_grad_(True)
            for param in self.generator.parameters():
                param.requires_grad_(False)
            with autocast(self.autocast_device_type, dtype=torch.bfloat16):
                d_stats = self.model(**batch, phase="discriminator")
                if "loss" not in d_stats:
                    raise KeyError("model output dict must contain 'loss' key for discriminator phase")
                d_loss = d_stats["loss"] / self.args.grad_accum_steps
            self.scaler.scale(d_loss).backward()

            if step % self.args.grad_accum_steps == 0:
                self.scaler.unscale_(self.generator_optimizer)
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=self.args.max_norm)
                self.scaler.step(self.generator_optimizer)

                self.scaler.unscale_(self.discriminator_optimizer)
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=self.args.max_norm)
                self.scaler.step(self.discriminator_optimizer)

                self.scaler.update()
                self.generator_optimizer.zero_grad(set_to_none=True)
                self.discriminator_optimizer.zero_grad(set_to_none=True)

            if step % self.args.log_steps == 0:
                g_msg = f"generator lr: {self.generator_scheduler.get_last_lr()[0]:.6f}, "
                for k, v in g_stats.items():
                    g_msg += f"{k}: {v.item():.3f}, "
                tqdm.write(g_msg)

                d_msg = f"discriminator lr: {self.discriminator_scheduler.get_last_lr()[0]:.6f}, "
                for k, v in d_stats.items():
                    d_msg += f"{k}: {v.item():.3f}, "
                tqdm.write(d_msg)

            g_stats_list.append(tensor_to_float_stats(g_stats))
            d_stats_list.append(tensor_to_float_stats(d_stats))

        self.generator_scheduler.step()
        self.discriminator_scheduler.step()

        g_avg_stats = average_stats(g_stats_list)
        d_avg_stats = average_stats(d_stats_list)

        return g_avg_stats, d_avg_stats

    @torch.inference_mode()
    def validate_epoch(self) -> tuple[dict[str, float], dict[str, float]]:
        g_stats_list = []
        d_stats_list = []
        self.model.eval()
        for batch in tqdm(
            self.valid_dataloader, desc="validating", total=len(self.valid_dataloader), leave=False, dynamic_ncols=True
        ):
            batch = {k: v.to(self.args.device) if torch.is_tensor(v) else v for k, v in batch.items()}
            g_stats = self.model(**batch, phase="generator")
            d_stats = self.model(**batch, phase="discriminator")
            g_stats_list.append(tensor_to_float_stats(g_stats))
            d_stats_list.append(tensor_to_float_stats(d_stats))

        g_stats_avg = average_stats(g_stats_list)
        d_stats_avg = average_stats(d_stats_list)

        return g_stats_avg, d_stats_avg

    def save_checkpoint(self, epoch: int):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "generator_optimizer_state_dict": self.generator_optimizer.state_dict(),
            "discriminator_optimizer_state_dict": self.discriminator_optimizer.state_dict(),
            "generator_scheduler_state_dict": self.generator_scheduler.state_dict(),
            "discriminator_scheduler_state_dict": self.discriminator_scheduler.state_dict(),
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
        self.generator_optimizer.load_state_dict(checkpoint["generator_optimizer_state_dict"])
        self.discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer_state_dict"])
        self.generator_scheduler.load_state_dict(checkpoint["generator_scheduler_state_dict"])
        self.discriminator_scheduler.load_state_dict(checkpoint["discriminator_scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.best_valid_loss = checkpoint["best_valid_loss"]
        self.start_epoch = checkpoint["epoch"]

    def train(self):
        self.args.out_dir.mkdir(exist_ok=True)
        for epoch in tqdm(range(self.start_epoch, self.args.epochs), desc="Training", dynamic_ncols=True):
            train_g_stats, train_d_stats = self.train_epoch()
            valid_g_stats, valid_d_stats = self.validate_epoch()
            tqdm.write(
                f"epoch {epoch + 1}/{self.args.epochs}, "
                f"train generator loss: {train_g_stats['loss']:.3f}, "
                f"train discriminator loss: {train_d_stats['loss']:.3f}, "
                f"valid generator loss: {valid_g_stats['loss']:.3f}, "
                f"valid discriminator loss: {valid_d_stats['loss']:.3f}"
            )

            valid_generator_loss = valid_g_stats["loss"]
            if valid_generator_loss < self.best_valid_loss:
                self.best_valid_loss = valid_generator_loss
                torch.save(self.model.state_dict(), self.args.out_dir / "best_model.pt")
            self.save_checkpoint(epoch + 1)

            self.stats[epoch + 1] = {
                "train_generator": train_g_stats,
                "train_discriminator": train_d_stats,
                "valid_generator": valid_g_stats,
                "valid_discriminator": valid_d_stats,
            }
            with open(self.args.out_dir / "stats.json", "w", encoding="utf-8") as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=4)
