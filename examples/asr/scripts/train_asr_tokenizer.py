import argparse
import os
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--vocab_size", type=int)
    args = parser.parse_args()
    os.makedirs(Path(args.out_dir), exist_ok=True)
    trainer = BpeTrainer(vocab_size=args.vocab_size, special_tokens=["[BLANK]", "[UNK]"])
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.train(files=[args.text_path], trainer=trainer)

    # save tokenizer
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        additional_special_tokens=["[BLANK]"],
    )
    fast_tokenizer.save_pretrained(args.out_dir)


if __name__ == "__main__":
    main()
