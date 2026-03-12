from pathlib import Path

import hydra
import torch
from model.conformer import ConformerRNNT
from omegaconf import DictConfig
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from slp.collate_fn.speech_to_text import SpeechToTextCollateFn
from slp.dataset.speech_text_dataset import SpeechTextDataset
from slp.modules.frontend.log_mel_spectrogram import LogMelSpectrogram
from slp.modules.frontend.spec_augment import SpecAugment
from slp.scheduler.linear_lr_scheduler import LinearLRScheduler
from slp.trainer import Trainer, TrainerArguments


@hydra.main(version_base=None)
def main(config: DictConfig):

    # dataset
    train_dataset = SpeechTextDataset(config.dataset.train_json_path)
    dev_dataset = SpeechTextDataset(config.dataset.valid_json_path)

    # model
    frontend = LogMelSpectrogram(
        fft_size=config.frontend.fft_size,
        hop_size=config.frontend.hop_size,
        window_size=config.frontend.window_size,
        mel_size=config.frontend.mel_size,
        sample_rate=config.frontend.sample_rate,
        min_freq=config.frontend.min_freq,
        max_freq=config.frontend.max_freq,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_dir)
    spec_augment = SpecAugment(
        num_freq_masks=config.frontend.num_freq_masks,
        num_time_masks=config.frontend.num_time_masks,
        max_freq_mask_size=config.frontend.max_freq_mask_size,
        max_time_mask_size=config.frontend.max_time_mask_size,
    )
    model = ConformerRNNT(
        vocab_size=config.tokenizer.vocab_size,
        input_size=config.frontend.mel_size,
        hidden_size=config.model.hidden_size,
        num_heads=config.model.num_heads,
        kernel_size=config.model.kernel_size,
        num_blocks=config.model.num_blocks,
        num_layers=config.model.num_layers,
        pred_size=config.model.pred_size,
        joint_size=config.model.joint_size,
        dropout_rate=config.model.dropout_rate,
        blank_token_id=tokenizer.convert_tokens_to_ids("[BLANK]"),
        ignore_token_id=config.model.ignore_token_id,
        min_chunk_size=config.model.min_chunk_size,
        max_chunk_size=config.model.max_chunk_size,
        streaming_mask_ratio=config.model.streaming_mask_ratio,
        ctc_loss_weight=config.model.ctc_loss_weight,
    )

    # dataloader
    collate_fn = SpeechToTextCollateFn(
        tokenizer=tokenizer,
        frontend=frontend,
        max_length=config.tokenizer.max_length,
        spec_augment=spec_augment,
        ignore_token_id=config.model.ignore_token_id,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train.dataloader.train_batch_size,
        shuffle=True,
        num_workers=config.train.dataloader.num_workers,
        pin_memory=config.train.dataloader.pin_memory,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        dev_dataset,
        batch_size=config.train.dataloader.valid_batch_size,
        shuffle=False,
        num_workers=config.train.dataloader.num_workers,
        pin_memory=config.train.dataloader.pin_memory,
        collate_fn=collate_fn,
    )

    # training
    optimizer = AdamW(
        model.parameters(), lr=config.train.optimizer.lr, weight_decay=config.train.optimizer.weight_decay
    )
    scheduler = LinearLRScheduler(
        optimizer,
        total_steps=(len(train_dataloader) // config.train.grad_accum_steps) * config.train.epochs,
        warmup_steps=config.train.scheduler.warmup_steps,
    )
    scaler = GradScaler("cuda" if torch.cuda.is_available() else "cpu")
    args = TrainerArguments(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        out_dir=Path(config.train.out_dir),
        epochs=config.train.epochs,
        checkpoint_path=Path(config.train.checkpoint_path) if config.train.checkpoint_path else None,
        grad_accum_steps=config.train.grad_accum_steps,
        max_norm=config.train.max_norm,
        log_steps=config.train.log_steps,
    )
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        args=args,
    )
    trainer.train()


if __name__ == "__main__":
    main()
