import logging
import os

import evaluate
import hydra
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


@hydra.main(version_base=None)
def main(config: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    test_dataset = load_dataset("wmt/wmt14", "de-en", split="test", cache_dir=config.dataset.data_dir)

    # model
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(config.evaluate.model_path).to(device)  # type: ignore[arg-type]
    model.eval()

    hyp_list, ref_list = [], []
    os.makedirs(config.evaluate.out_dir, exist_ok=True)
    with (
        open(f"{config.evaluate.out_dir}/ref.txt", "w", encoding="utf-8") as f_ref,
        open(f"{config.evaluate.out_dir}/hyp.txt", "w", encoding="utf-8") as f_hyp,
    ):
        total = len(test_dataset)
        for i in tqdm(range(total), desc="Evaluating"):
            sample = test_dataset[i]
            ref = sample["translation"]["en"]
            src = sample["translation"]["de"]
            ref_list.append(ref)
            f_ref.write(ref + "\n")

            messages = [
                {"role": "system", "content": config.model.system_prompt},
                {"role": "user", "content": src},
            ]
            # training data does not contain <think> token, so we disable thinking
            messages_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            input_ids = tokenizer(messages_text, return_tensors="pt").input_ids
            with torch.no_grad():
                output = model.generate(input_ids.to(device), num_beams=config.evaluate.beam_size)  # type: ignore[attr-defined]
            hyp_text: str = tokenizer.decode(output[0])  # type: ignore[assignment]
            hyp_text = hyp_text[len(messages_text) :].replace(tokenizer.eos_token, "").strip()
            hyp_list.append(hyp_text)
            f_hyp.write(hyp_text + "\n")

            tqdm.write(f"[{i + 1}/{total}] ref: {ref}")
            tqdm.write(f"[{i + 1}/{total}] hyp: {hyp_text}")
    bleu = evaluate.load("sacrebleu")
    metric = bleu.compute(predictions=hyp_list, references=[[ref] for ref in ref_list])
    bleu_score = metric["score"]
    logger.info(f"bleu: {bleu_score:.5f}")
    with open(f"{config.evaluate.out_dir}/metric.txt", "w", encoding="utf-8") as f:
        f.write(f"bleu: {bleu_score:.5f}\n")


if __name__ == "__main__":
    main()
