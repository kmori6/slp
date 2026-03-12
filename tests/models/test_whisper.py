import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from slp.models.whisper import Whisper


def test_whisper(
    model_name: str = "openai/whisper-large-v3-turbo",
    sequence_length: int = 100,
    rtol: float = 1e-4,
    atol: float = 1e-4,
):
    processor = WhisperProcessor.from_pretrained(model_name)
    src_model = WhisperForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True, use_safetensors=True, device_map="cpu"
    )
    model = Whisper.load_pretrained(model_name)
    src_model.eval()
    model.eval()
    length = 16000 * 30
    speech = torch.randn(length)
    input_feats = processor(speech, sampling_rate=16000, return_tensors="pt").input_features
    input_features = input_feats.transpose(1, 2)
    attention_mask = torch.ones(1, input_features.shape[1])
    decoder_input_ids = torch.arange(sequence_length)[None, :]
    decoder_attention_mask = torch.ones_like(decoder_input_ids)

    with torch.no_grad():
        x = src_model(
            input_feats,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        ).logits
        y = model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask.bool(),
            decoder_attention_mask=decoder_attention_mask,
        )

    torch.testing.assert_close(x, y, rtol=rtol, atol=atol)
