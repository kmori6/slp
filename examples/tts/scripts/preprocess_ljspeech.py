import argparse
import json
import os
from pathlib import Path

import torchaudio
from phonemizer import phonemize
from tqdm import tqdm
from whisper.normalizers import EnglishTextNormalizer


def normalize_and_phonemize(text: str, normalizer: EnglishTextNormalizer) -> str:
    normalized_text = normalizer(text.strip())
    phoneme = phonemize(
        normalized_text,
        language="en-us",
        strip=True,
        preserve_punctuation=True,
        with_stress=True,
        backend="espeak",
    )
    return phoneme


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    normalizer = EnglishTextNormalizer()
    text_list = []
    with open(Path(args.data_dir) / "LJSpeech-1.1" / "metadata.csv", "r", encoding="utf-8") as f:
        lines = f.readlines()
    subsets = {"train": lines[:12600], "valid": lines[12600:12850], "test": lines[12850:]}
    for key in subsets.keys():
        dic_list = []
        for sample in tqdm(subsets[key], desc=f"Processing {key} set"):
            utt_id, _, text = sample.split("|")
            phoneme = normalize_and_phonemize(text, normalizer)
            audio_path = Path(args.data_dir) / "LJSpeech-1.1" / "wavs" / f"{utt_id}.wav"
            audio_info = torchaudio.info(audio_path)
            assert audio_info.num_channels == 1
            assert audio_info.sample_rate == 22050
            audio_length = audio_info.num_frames
            dic = {"text": phoneme, "audio_path": str(audio_path), "audio_length": audio_length, "utt_id": utt_id}
            dic_list.append(dic)
            if key == "train":
                text_list.append(phoneme + "\n")
        os.makedirs(Path(args.out_dir), exist_ok=True)
        with open(Path(args.out_dir) / f"{key}.json", "w", encoding="utf-8") as f:
            json.dump(dic_list, f, ensure_ascii=False, indent=4)

    # train text file for tokenizer
    text_file_path = Path(args.out_dir) / "train.txt"
    with open(text_file_path, "w", encoding="utf-8") as f:
        f.writelines(text_list)


if __name__ == "__main__":
    main()
