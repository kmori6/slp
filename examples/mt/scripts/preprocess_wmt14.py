import argparse
import json
from collections.abc import Callable
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer

SPLITS = ("train", "validation", "test")
SPLIT_TO_OUTPUT_NAME = {"validation": "valid"}
TRAIN_SPLIT = "train"


def make_normalizer(lang: str) -> Callable[[str], str]:
    if lang == "en":
        return EnglishTextNormalizer()
    return BasicTextNormalizer()


def normalize_translation_pair(
    translation: dict[str, str],
    src_lang: str,
    tgt_lang: str,
    src_normalizer: Callable[[str], str],
    tgt_normalizer: Callable[[str], str],
) -> tuple[str, str] | None:
    src_text = src_normalizer(translation[src_lang].strip())
    tgt_text = tgt_normalizer(translation[tgt_lang].strip())
    if not src_text or not tgt_text:
        return None
    return src_text, tgt_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--src_lang", type=str, default="de")
    parser.add_argument("--tgt_lang", type=str, default="en")
    args = parser.parse_args()

    src_normalizer = make_normalizer(args.src_lang)
    tgt_normalizer = make_normalizer(args.tgt_lang)
    dataset = load_dataset("wmt/wmt14", "de-en")
    train_text_lines: list[str] = []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        records: list[dict[str, str]] = []
        data = dataset[split]

        for idx, sample in enumerate(tqdm(data, desc=f"Processing {split}")):
            translation = sample["translation"]
            normalized_pair = normalize_translation_pair(
                translation=translation,
                src_lang=args.src_lang,
                tgt_lang=args.tgt_lang,
                src_normalizer=src_normalizer,
                tgt_normalizer=tgt_normalizer,
            )
            if normalized_pair is None:
                continue
            src_text, tgt_text = normalized_pair

            record = {"id": f"{split}_{idx}", "src_text": src_text, "tgt_text": tgt_text}
            records.append(record)

            if split == TRAIN_SPLIT:
                train_text_lines.extend((f"{src_text}\n", f"{tgt_text}\n"))

        split_name = SPLIT_TO_OUTPUT_NAME.get(split, split)
        json_path = out_dir / f"{split_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=4)

    text_file_path = out_dir / "train.txt"
    with open(text_file_path, "w", encoding="utf-8") as f:
        f.writelines(train_text_lines)


if __name__ == "__main__":
    main()
