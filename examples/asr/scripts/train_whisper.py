import hydra
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    WhisperForConditionalGeneration,
)

from slp.collate_fn.whisper import WhisperCollateFn
from slp.dataset.speech_text_dataset import SpeechTextDataset


@hydra.main(version_base=None)
def main(config: DictConfig):

    # dataset
    train_dataset = SpeechTextDataset(config.dataset.train_json_path)
    dev_dataset = SpeechTextDataset(config.dataset.valid_json_path)

    # model
    frontend = AutoFeatureExtractor.from_pretrained(config.model.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    tokenizer.set_prefix_tokens(language="english", task="transcribe")
    base_model = WhisperForConditionalGeneration.from_pretrained(config.model.model_name)

    # lora
    lora_config = LoraConfig(
        r=config.model.lora_rank,
        lora_alpha=config.model.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=config.model.lora_dropout,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # training
    collate_fn = WhisperCollateFn(
        tokenizer=tokenizer,
        frontend=frontend,
        max_length=base_model.config.max_target_positions,
        pad_token_id=tokenizer.pad_token_id,
    )
    training_args = TrainingArguments(
        output_dir=config.train.out_dir,
        eval_strategy="epoch",
        per_device_train_batch_size=config.train.dataloader.train_batch_size,
        per_device_eval_batch_size=config.train.dataloader.valid_batch_size,
        gradient_accumulation_steps=config.train.grad_accum_steps,
        optim=config.train.optimizer.type,
        learning_rate=config.train.optimizer.lr,
        weight_decay=config.train.optimizer.weight_decay,
        adam_beta1=config.train.optimizer.beta1,
        adam_beta2=config.train.optimizer.beta2,
        adam_epsilon=config.train.optimizer.eps,
        max_grad_norm=config.train.optimizer.max_grad_norm,
        num_train_epochs=config.train.epochs,
        lr_scheduler_type=config.train.scheduler.type,
        warmup_steps=config.train.scheduler.warmup_steps,
        logging_steps=config.train.log_steps,
        save_strategy="epoch",
        bf16=True,
        torch_compile=config.train.torch_compile,
        dataloader_num_workers=config.train.dataloader.num_workers,
        dataloader_pin_memory=config.train.dataloader.pin_memory,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collate_fn,
    )
    trainer.train()


if __name__ == "__main__":
    main()
