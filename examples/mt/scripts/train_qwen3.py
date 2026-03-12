import hydra
from datasets import load_dataset
from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


@hydra.main(version_base=None)
def main(config: DictConfig):

    # dataset
    train_dataset = load_dataset("wmt/wmt14", "de-en", split="train", cache_dir=config.dataset.data_dir)
    dev_dataset = load_dataset("wmt/wmt14", "de-en", split="validation", cache_dir=config.dataset.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)

    def preprocess_function(example):
        messages = [
            {"role": "system", "content": config.model.system_prompt},
            {"role": "user", "content": example["translation"]["de"]},
            {"role": "assistant", "content": example["translation"]["en"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        tokenized = tokenizer(text, max_length=config.model.max_length, truncation=True)
        return tokenized

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

    # model
    base_model = AutoModelForCausalLM.from_pretrained(config.model.model_name)

    # lora
    lora_config = LoraConfig(
        r=config.model.lora_rank,
        lora_alpha=config.model.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=config.model.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    # training
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
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()


if __name__ == "__main__":
    main()
