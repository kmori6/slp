import torch
from transformers import AutoModelForCausalLM

from slp.models.qwen3 import Qwen3


def test_qwen3(model_name: str = "Qwen/Qwen3-0.6B", sequence_length: int = 100, rtol: float = 1e-4, atol: float = 1e-4):
    src_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map="cpu")
    model = Qwen3.load_pretrained(model_name)
    src_model.eval()
    model.eval()
    input_ids = torch.arange(sequence_length)[None, :]
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        x = src_model(input_ids, attention_mask=attention_mask).logits
        y = model(input_ids, torch.ones(1, sequence_length, sequence_length, dtype=torch.bool).tril())

    torch.testing.assert_close(x, y, rtol=rtol, atol=atol)
