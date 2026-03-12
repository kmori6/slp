import argparse
import os
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer for TTS phoneme text.")
    parser.add_argument("--text_path", type=str, required=True, help="Path to training text file.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save tokenizer files.")
    parser.add_argument("--vocab_size", type=int, default=1024, help="Vocabulary size.")
    args = parser.parse_args()
    os.makedirs(Path(args.out_dir), exist_ok=True)

    special_tokens = ["[UNK]", "[PAD]"]
    trainer = BpeTrainer(vocab_size=args.vocab_size, special_tokens=special_tokens)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train(files=[args.text_path], trainer=trainer)
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    fast_tokenizer.save_pretrained(args.out_dir)


if __name__ == "__main__":
    main()
