from pathlib import Path

import hydra
import torch
import torch.optim as optim
from models.vits import Vits
from omegaconf import DictConfig
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from slp.adversarial_trainer import AdversarialTrainer, AdversarialTrainerArguments
from slp.collate_fn.text_to_speech import TextToSpeechCollateFn
from slp.dataset.speech_text_dataset import SpeechTextDataset
from slp.modules.frontend.linear_spectrogram import LinearSpectrogram
from slp.scheduler.epoch_decay_lr_scheduler import EpochDecayLRScheduler
from slp.utils.seed import fix_seed


@hydra.main(version_base=None, config_path="../config", config_name="vits")
def main(config: DictConfig):
    # FIXME: use_deterministic_algorithms=False for now, because of F.pad() and stft() raise Error.
    fix_seed(config.train.seed, use_deterministic_algorithms=False)

    train_dataset = SpeechTextDataset(config.dataset.train_json_path, sample_rate=config.dataset.sample_rate)
    valid_dataset = SpeechTextDataset(config.dataset.valid_json_path, sample_rate=config.dataset.sample_rate)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_dir)
    frontend = LinearSpectrogram(
        fft_size=config.model.fft_size,
        hop_size=config.model.hop_size,
        window_size=config.model.window_size,
    )
    collate_fn = TextToSpeechCollateFn(
        tokenizer=tokenizer,
        frontend=frontend,
        max_length=config.model.max_length,
        segment_size=config.model.segment_size,
        pad_token_id=tokenizer.pad_token_id,
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
        valid_dataset,
        batch_size=config.train.dataloader.valid_batch_size,
        shuffle=False,
        num_workers=config.train.dataloader.num_workers,
        pin_memory=config.train.dataloader.pin_memory,
        collate_fn=collate_fn,
    )

    model = Vits(
        vocab_size=len(tokenizer),
        spec_size=config.model.fft_size // 2 + 1,
        segment_size=config.model.segment_size,
        hidden_size=config.model.hidden_size,
        text_ffn_size=config.model.text_ffn_size,
        text_num_heads=config.model.text_num_heads,
        text_num_layers=config.model.text_num_layers,
        text_dropout_rate=config.model.text_dropout_rate,
        text_window_size=config.model.text_window_size,
        posterior_kernel_size=config.model.posterior_kernel_size,
        posterior_dilation_rate=config.model.posterior_dilation_rate,
        posterior_num_layers=config.model.posterior_num_layers,
        posterior_dropout_rate=config.model.posterior_dropout_rate,
        flow_kernel_size=config.model.flow_kernel_size,
        flow_dilation_rate=config.model.flow_dilation_rate,
        flow_num_blocks=config.model.flow_num_blocks,
        flow_num_layers=config.model.flow_num_layers,
        flow_dropout_rate=config.model.flow_dropout_rate,
        duration_kernel_size=config.model.duration_kernel_size,
        duration_dropout_rate=config.model.duration_dropout_rate,
        duration_num_flows_blocks=config.model.duration_num_flows_blocks,
        duration_num_prior_layers=config.model.duration_num_prior_layers,
        duration_num_condition_layers=config.model.duration_num_condition_layers,
        duration_num_posterior_layers=config.model.duration_num_posterior_layers,
        duration_spline_num_bins=config.model.duration_spline_num_bins,
        duration_spline_bound=config.model.duration_spline_bound,
        speaker_size=config.model.speaker_size,
        cond_size=config.model.cond_size,
        fft_size=config.model.fft_size,
        hop_size=config.model.hop_size,
        window_size=config.model.window_size,
        mel_size=config.model.mel_size,
        sample_rate=config.model.sample_rate,
        min_freq=config.model.min_freq,
        max_freq=config.model.max_freq,
        periods=config.model.periods,
        scales=config.model.scales,
        negative_slope=config.model.negative_slope,
        mel_loss_weight=config.model.mel_loss_weight,
        kl_loss_weight=config.model.kl_loss_weight,
        dur_loss_weight=config.model.dur_loss_weight,
        gen_adv_loss_weight=config.model.gen_adv_loss_weight,
        fm_loss_weight=config.model.fm_loss_weight,
        disc_adv_loss_weight=config.model.disc_adv_loss_weight,
    )
    generator = model.generator
    discriminator = model.discriminator

    generator_optimizer = optim.AdamW(
        generator.parameters(),
        lr=config.train.optimizer.generator.lr,
        betas=(config.train.optimizer.generator.beta1, config.train.optimizer.generator.beta2),
        eps=config.train.optimizer.generator.eps,
        weight_decay=config.train.optimizer.generator.weight_decay,
    )
    discriminator_optimizer = optim.AdamW(
        discriminator.parameters(),
        lr=config.train.optimizer.discriminator.lr,
        betas=(config.train.optimizer.discriminator.beta1, config.train.optimizer.discriminator.beta2),
        eps=config.train.optimizer.discriminator.eps,
        weight_decay=config.train.optimizer.discriminator.weight_decay,
    )
    generator_scheduler = EpochDecayLRScheduler(generator_optimizer, gamma=config.train.scheduler.generator.gamma)
    discriminator_scheduler = EpochDecayLRScheduler(
        discriminator_optimizer, gamma=config.train.scheduler.discriminator.gamma
    )
    scaler = GradScaler("cuda" if torch.cuda.is_available() else "cpu")
    args = AdversarialTrainerArguments(
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        out_dir=Path(config.train.out_dir),
        epochs=config.train.epochs,
        checkpoint_path=Path(config.train.checkpoint_path) if config.train.checkpoint_path else None,
        grad_accum_steps=config.train.grad_accum_steps,
        max_norm=config.train.max_grad_norm,
        log_steps=config.train.log_steps,
    )
    trainer = AdversarialTrainer(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator_scheduler=generator_scheduler,
        discriminator_scheduler=discriminator_scheduler,
        scaler=scaler,
        args=args,
    )
    trainer.train()


if __name__ == "__main__":
    main()
