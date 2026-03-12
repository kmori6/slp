from logging import getLogger
from pathlib import Path

import hydra
import torch
from evaluate import load
from omegaconf import DictConfig
from peft import AutoPeftModel
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoTokenizer

from slp.dataset.speech_text_dataset import SpeechTextDataset

logger = getLogger(__name__)


def recognize(
    frontend: AutoFeatureExtractor,
    model: AutoPeftModel,
    audio: torch.Tensor,
    beam_size: int,
    device: torch.device,
) -> list[int]:
    processed = frontend(audio, sampling_rate=16_000, return_tensors="pt")
    input_features = processed.input_features.to(device)
    input_features = input_features.to(model.dtype)
    predicted_ids = model.generate(
        input_features,
        num_beams=beam_size,
        use_cache=True,
        cache_implementation="static",
        language="english",
        task="transcribe",
    )
    return predicted_ids[0]


@hydra.main(version_base=None)
def main(config: DictConfig):

    # setup
    out_dir = Path(config.evaluate.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    test_dataset = SpeechTextDataset(config.dataset.test_json_path)

    # model
    frontend = AutoFeatureExtractor.from_pretrained(config.model.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    model = AutoPeftModel.from_pretrained(config.evaluate.model_path).to(device)

    hyp_list: list[str] = []
    ref_list: list[str] = []
    with (
        open(f"{config.evaluate.out_dir}/ref.txt", "w", encoding="utf-8") as f_ref,
        open(f"{config.evaluate.out_dir}/hyp.txt", "w", encoding="utf-8") as f_hyp,
    ):
        total = len(test_dataset)
        for i in tqdm(range(total), desc="Evaluating"):
            sample = test_dataset[i]
            ref_list.append(sample["text"])
            f_ref.write(sample["text"] + "\n")

            hyp = recognize(frontend, model, sample["speech"], config.evaluate.beam_size, device)
            text: str = tokenizer.decode(hyp, skip_special_tokens=True)  # type: ignore[assignment]
            hyp_list.append(text)
            f_hyp.write(text + "\n")

            tqdm.write(f"[{i + 1}/{total}] ref: {sample['text']}")
            tqdm.write(f"[{i + 1}/{total}] hyp: {text}")
    wer = load("wer")
    metric = wer.compute(predictions=hyp_list, references=ref_list)
    logger.info(f"wer: {metric:.5f}")
    with open(f"{config.evaluate.out_dir}/metric.txt", "w", encoding="utf-8") as f:
        f.write(f"wer: {metric:.5f}")


if __name__ == "__main__":
    main()
