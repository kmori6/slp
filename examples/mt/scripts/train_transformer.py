from pathlib import Path

import hydra
import torch
from datasets import load_dataset
from models.transformer import Transformer
from omegaconf import DictConfig
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from slp.collate_fn.text_to_text import TextToTextCollateFn
from slp.scheduler.linear_lr_scheduler import LinearLRScheduler
from slp.trainer import Trainer, TrainerArguments


@hydra.main(version_base=None)
def main(config: DictConfig):

    # dataset
    train_dataset = load_dataset("wmt/wmt14", "de-en", split="train", cache_dir=config.dataset.data_dir)
    dev_dataset = load_dataset("wmt/wmt14", "de-en", split="validation", cache_dir=config.dataset.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_dir)

    # preprocess
    def preprocess_function(example):
        src_text = example["translation"]["de"]
        tgt_text = example["translation"]["en"]
        model_inputs = tokenizer(src_text, max_length=config.model.max_length, truncation=True)
        labels = tokenizer(tgt_text, max_length=config.model.max_length, truncation=True)
        model_inputs["decoder_input_ids"] = labels["input_ids"]
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["decoder_attention_mask"] = labels["attention_mask"]
        return model_inputs

    train_dataset = train_dataset.map(
        preprocess_function,
        remove_columns=train_dataset.column_names,
        num_proc=config.dataset.num_proc,
        keep_in_memory=True,
    )
    dev_dataset = dev_dataset.map(
        preprocess_function,
        remove_columns=dev_dataset.column_names,
        num_proc=config.dataset.num_proc,
        keep_in_memory=True,
    )

    # dataloader
    collate_fn = TextToTextCollateFn(tokenizer.pad_token_id, ignore_token_id=config.model.ignore_token_id)
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

    # model
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config.model.d_model,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        d_ff=config.model.d_ff,
        dropout_rate=config.model.dropout_rate,
        max_length=config.model.max_length,
        label_smoothing=config.model.label_smoothing,
        ignore_token_id=config.model.ignore_token_id,
    )
    model = torch.compile(model)  # type: ignore[assignment]

    # training
    optimizer = AdamW(
        model.parameters(),  # type: ignore[attr-defined]
        lr=config.train.optimizer.lr,
        weight_decay=config.train.optimizer.weight_decay,
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
        max_norm=config.train.optimizer.max_grad_norm,
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
