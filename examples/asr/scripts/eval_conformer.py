from logging import getLogger

import hydra
import torch
from evaluate import load
from model.conformer import ConformerRNNT
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoTokenizer

from slp.dataset.speech_text_dataset import SpeechTextDataset
from slp.inference.audio_chunker import AudioChunker
from slp.modules.frontend.log_mel_spectrogram import LogMelSpectrogram
from slp.modules.search.transducer_beam_search import TransducerBeamSearch

logger = getLogger(__name__)


def _encode_waveform(
    frontend: LogMelSpectrogram,
    model: ConformerRNNT,
    waveform: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    waveform = waveform.to(device)
    length = torch.tensor([waveform.shape[-1]], dtype=torch.long, device=device)
    feat, mask = frontend(waveform, length)
    enc_out, _ = model.encode(feat, mask)
    return enc_out


def recognize(
    frontend: LogMelSpectrogram,
    model: ConformerRNNT,
    audio: torch.Tensor,
    streamer: AudioChunker,
    searcher: TransducerBeamSearch,
    chunk_size: int,
    left_chunk_size: int,
    device: torch.device,
    streaming: bool = True,
) -> list[int]:
    searcher.reset()
    batched_audio = audio.unsqueeze(0)

    if not streaming:
        searcher.search(_encode_waveform(frontend, model, batched_audio, device))
        return searcher.best_token_ids()

    center_start = left_chunk_size
    center_end = left_chunk_size + chunk_size

    for chunk in streamer.stream(batched_audio):
        enc_out = _encode_waveform(frontend, model, chunk.audio, device)
        # Trim left/right context frames, keep only center chunk_size frames.
        searcher.search(enc_out[:, center_start:center_end, :])

    return searcher.best_token_ids()


@hydra.main(version_base=None)
def main(config: DictConfig):
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

    # dataset
    test_dataset = SpeechTextDataset(config.dataset.test_json_path)

    # model
    frontend = LogMelSpectrogram(
        fft_size=config.frontend.fft_size,
        hop_size=config.frontend.hop_size,
        window_size=config.frontend.window_size,
        mel_size=config.frontend.mel_size,
        sample_rate=config.frontend.sample_rate,
        min_freq=config.frontend.min_freq,
        max_freq=config.frontend.max_freq,
    )
    streamer = AudioChunker(
        chunk_ms=config.evaluate.chunk_ms,
        left_context_ms=config.evaluate.left_chunk_ms,
        right_context_ms=config.evaluate.right_chunk_ms,
        sample_rate=config.frontend.sample_rate,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_dir)
    model = ConformerRNNT(
        vocab_size=config.tokenizer.vocab_size,
        input_size=config.frontend.mel_size,
        hidden_size=config.model.hidden_size,
        num_heads=config.model.num_heads,
        kernel_size=config.model.kernel_size,
        num_blocks=config.model.num_blocks,
        num_layers=config.model.num_layers,
        pred_size=config.model.pred_size,
        joint_size=config.model.joint_size,
        dropout_rate=config.model.dropout_rate,
        blank_token_id=tokenizer.convert_tokens_to_ids("[BLANK]"),
        ignore_token_id=config.model.ignore_token_id,
        min_chunk_size=config.model.min_chunk_size,
        max_chunk_size=config.model.max_chunk_size,
        streaming_mask_ratio=config.model.streaming_mask_ratio,
        ctc_loss_weight=config.model.ctc_loss_weight,
    )
    state_dict = torch.load(config.evaluate.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    searcher = TransducerBeamSearch(
        prediction_network=model.prediction_network,
        joint_network=model.joint_network,
        beam_width=config.evaluate.beam_size,
        blank_token_id=model.blank_token_id,
    )

    hyp_list, ref_list = [], []
    with (
        open(f"{config.evaluate.out_dir}/ref.txt", "w", encoding="utf-8") as f_ref,
        open(f"{config.evaluate.out_dir}/hyp.txt", "w", encoding="utf-8") as f_hyp,
    ):
        total = len(test_dataset)
        for i in tqdm(range(total), desc="Evaluating"):
            sample = test_dataset[i]
            ref_list.append(sample.text)
            f_ref.write(sample.text + "\n")
            hyp = recognize(
                frontend=frontend,
                model=model,
                audio=sample.speech,
                streamer=streamer,
                searcher=searcher,
                chunk_size=config.evaluate.chunk_size,
                left_chunk_size=config.evaluate.left_chunk_size,
                device=device,
                streaming=config.evaluate.streaming,
            )
            text: str = tokenizer.decode(hyp, skip_special_tokens=True)  # type: ignore[assignment]
            text = text.strip()
            hyp_list.append(text)
            f_hyp.write(text + "\n")

            tqdm.write(f"[{i + 1}/{total}] ref: {sample.text}")
            tqdm.write(f"[{i + 1}/{total}] hyp: {text}")
    wer = load("wer")
    metric = wer.compute(predictions=hyp_list, references=ref_list)
    logger.info(f"wer: {metric:.5f}")
    with open(f"{config.evaluate.out_dir}/metric.txt", "w", encoding="utf-8") as f:
        f.write(f"wer: {metric:.5f}")


if __name__ == "__main__":
    main()
