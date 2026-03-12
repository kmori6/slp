from logging import getLogger
from pathlib import Path

import hydra
import torch
from datasets import load_dataset
from evaluate import load
from models.transformer import Transformer
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoTokenizer

from slp.modules.search.beam_search import BeamSearch

logger = getLogger(__name__)


def translate(
    model: Transformer,
    tokenizer: AutoTokenizer,
    searcher: BeamSearch,
    src_text: str,
    beam_size: int,
    device: torch.device,
) -> str:
    inputs = tokenizer(src_text, return_tensors="pt").to(device)
    x_enc = model.encode(inputs["input_ids"], inputs["attention_mask"])
    hyp = searcher.search(
        x_enc=x_enc,
        beam_size=beam_size,
        length_buffer=50,
        length_penalty=0.6,
    )
    # Skip BOS token when decoding.
    text: str = tokenizer.decode(hyp.token[1:], skip_special_tokens=True)  # type: ignore[assignment]
    return text.strip()


@hydra.main(version_base=None)
def main(config: DictConfig):
    out_dir = Path(config.evaluate.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    test_dataset = load_dataset("wmt/wmt14", "de-en", split="test", cache_dir=config.dataset.data_dir)

    # model
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_dir)
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
    ).to(device)
    model = torch.compile(model)  # type: ignore[assignment]
    state_dict = torch.load(config.evaluate.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.eval()

    searcher = BeamSearch(
        decoder=model,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    hyp_list, ref_list = [], []
    with (
        open(out_dir / "ref.txt", "w", encoding="utf-8") as f_ref,
        open(out_dir / "hyp.txt", "w", encoding="utf-8") as f_hyp,
    ):
        total = len(test_dataset)
        for i in tqdm(range(total), desc="Evaluating"):
            sample = test_dataset[i]
            ref = sample["translation"]["en"]
            src = sample["translation"]["de"]
            ref_list.append(ref)
            f_ref.write(ref + "\n")
            text = translate(
                model=model,
                tokenizer=tokenizer,
                searcher=searcher,
                src_text=src,
                beam_size=config.evaluate.beam_size,
                device=device,
            )
            hyp_list.append(text)
            f_hyp.write(text + "\n")
            tqdm.write(f"[{i + 1}/{total}] ref: {ref}")
            tqdm.write(f"[{i + 1}/{total}] hyp: {text}")

    bleu = load("sacrebleu")
    metric = bleu.compute(predictions=hyp_list, references=[[ref] for ref in ref_list])
    bleu_score = metric["score"]
    logger.info(f"bleu: {bleu_score:.5f}")
    with open(out_dir / "metric.txt", "w", encoding="utf-8") as f:
        f.write(f"bleu: {bleu_score:.5f}\n")


if __name__ == "__main__":
    main()
