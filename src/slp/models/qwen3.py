import re

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from slp.modules.transformer.encoder import Encoder
from slp.modules.transformer.positional_encoding import PositionalEncoding
from slp.modules.transformer.pre_norm_encoder_layer import PreNormEncoderLayer
from slp.modules.transformer.rotary_positional_multi_head_attention import RotaryPositionalMultiHeadAttention
from slp.modules.transformer.swiglu_feed_forward import SwiGLUFeedForward
from slp.modules.transformer.types import Hypothesis
from slp.utils.mask import causal_mask

TRANSFORMERS_KEY_MAP = {
    r"model\.embed_tokens\.weight": "token_embed.weight",
    r"model\.layers\.(\d+)\.self_attn\.q_proj\.weight": "decoder.layers.{layer_id}.mha.w_q.weight",
    r"model\.layers\.(\d+)\.self_attn\.q_proj\.bias": "decoder.layers.{layer_id}.mha.w_q.bias",
    r"model\.layers\.(\d+)\.self_attn\.q_norm\.weight": "decoder.layers.{layer_id}.mha.n_q.weight",
    r"model\.layers\.(\d+)\.self_attn\.k_proj\.weight": "decoder.layers.{layer_id}.mha.w_k.weight",
    r"model\.layers\.(\d+)\.self_attn\.k_proj\.bias": "decoder.layers.{layer_id}.mha.w_k.bias",
    r"model\.layers\.(\d+)\.self_attn\.k_norm\.weight": "decoder.layers.{layer_id}.mha.n_k.weight",
    r"model\.layers\.(\d+)\.self_attn\.v_proj\.weight": "decoder.layers.{layer_id}.mha.w_v.weight",
    r"model\.layers\.(\d+)\.self_attn\.v_proj\.bias": "decoder.layers.{layer_id}.mha.w_v.bias",
    r"model\.layers\.(\d+)\.self_attn\.o_proj\.weight": "decoder.layers.{layer_id}.mha.w_o.weight",
    r"model\.layers\.(\d+)\.mlp\.gate_proj\.weight": "decoder.layers.{layer_id}.ffn.w_1.weight",
    r"model\.layers\.(\d+)\.mlp\.up_proj\.weight": "decoder.layers.{layer_id}.ffn.w_v.weight",
    r"model\.layers\.(\d+)\.mlp\.down_proj\.weight": "decoder.layers.{layer_id}.ffn.w_2.weight",
    r"model\.layers\.(\d+)\.input_layernorm\.weight": "decoder.layers.{layer_id}.layer_norm1.weight",
    r"model\.layers\.(\d+)\.post_attention_layernorm\.weight": "decoder.layers.{layer_id}.layer_norm2.weight",
    r"model\.norm\.weight": "layer_norm.weight",
    r"lm_head\.weight": "linear.weight",
}


# https://github.com/huggingface/transformers/issues/25199
# https://github.com/huggingface/transformers/issues/33826
def permute_weights(weights: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Permute weights for the original interleave implementation from sliced rotary"""
    odim, idim = weights.shape
    weights = weights.view(num_heads, 2, odim // num_heads // 2, idim)
    weights = weights.transpose(1, 2)
    weights = weights.reshape(odim, idim)
    return weights


def permute_bias(bias: torch.Tensor, num_heads: int) -> torch.Tensor:
    odim = bias.shape[0]
    bias = bias.view(num_heads, 2, odim // num_heads // 2)
    bias = bias.transpose(1, 2)
    bias = bias.reshape(odim)
    return bias


def permute_norm(weights: torch.Tensor) -> torch.Tensor:
    odim = weights.shape[0]
    weights = weights.view(2, odim // 2)
    weights = weights.transpose(0, 1)
    weights = weights.reshape(odim)
    return weights


class Qwen3MultiHeadAttention(RotaryPositionalMultiHeadAttention):
    def __init__(
        self,
        d_model: int,
        d_k: int,
        num_heads: int,
        num_kv_heads: int,
        dropout_rate: float,
        eps: float,
        query_bias: bool = False,
        key_bias: bool = False,
        value_bias: bool = False,
    ):
        super().__init__(d_model, d_k, num_heads, num_kv_heads, dropout_rate, query_bias, key_bias, value_bias)
        self.n_q = nn.RMSNorm(d_k, eps=eps)
        self.n_k = nn.RMSNorm(d_k, eps=eps)

    def _forward_qkv(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Args:
            q (torch.Tensor): Query tensor (batch_size, target_sequence_length, hidden_size).
            k (torch.Tensor): Key tensor (batch_size, source_sequence_length, hidden_size).
            v (torch.Tensor): Value tensor (batch_size, source_sequence_length, hidden_size).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of query, key, and value tensors
                - query: (batch_size, num_heads, target_sequence_length, d_k),
                - key: (batch_size, num_key_value_heads, source_sequence_length, d_k),
                - value: (batch_size, num_key_value_heads, source_sequence_length, d_k).
        """
        b, t, s = q.shape[0], q.shape[1], k.shape[1]
        q = self.n_q(self.w_q(q).view(b, t, self.h, self.d_k)).transpose(1, 2)  # (b, h, t, d_k)
        k = self.n_k(self.w_k(k).view(b, s, self.h_kv, self.d_k)).transpose(1, 2)  # (b, h, s, d_k)
        v = self.w_v(v).view(b, s, self.h_kv, self.d_k).transpose(1, 2)  # (b, h, s, d_k)
        return q, k, v


class Qwen3(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_k: int,
        num_heads: int,
        num_kv_heads: int,
        d_ff: int,
        num_layers: int,
        dropout_rate: float,
        max_length: int,
        base: float,
        eps: float,
        eos_token_id: int,
    ):
        super().__init__()
        self.eos_token_id = eos_token_id
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.position_embed = PositionalEncoding(d_k, max_length, base=base)
        self.decoder = Encoder(
            encoder_layer=PreNormEncoderLayer(
                mha_module=Qwen3MultiHeadAttention(d_model, d_k, num_heads, num_kv_heads, dropout_rate, eps=eps),
                mha_norm_module=nn.RMSNorm(d_model, eps=eps),
                ffn_module=SwiGLUFeedForward(d_model, d_ff, dropout_rate, bias=False),
                ffn_norm_module=nn.RMSNorm(d_model, eps=eps),
                dropout_rate=dropout_rate,
            ),
            num_layers=num_layers,
        )
        self.layer_norm = nn.RMSNorm(d_model, eps=eps)
        self.linear = nn.Linear(d_model, vocab_size, bias=False)
        # tie weights
        self.linear.weight = self.token_embed.weight

    def forward(self, token: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """

        Args:
            token (torch.Tensor): Token tensor (batch_size, sequence_length).
            mask (torch.Tensor): Sequence mask tensor (batch_size, sequence_length, sequence_length).

        Returns:
            torch.Tensor: Output tensor (batch_size, sequence_length, vocab_size).
        """
        x = self.token_embed(token)  # (batch_size, sequence_length, d_model)
        p = self.position_embed(x)  # (1, sequence_length, d_model // num_heads)
        x = self.decoder(x, mask, p_q=p, p_k=p)
        x = self.layer_norm(x)
        x = self.linear(x)
        return x

    @torch.inference_mode()
    def predict(self, token: torch.Tensor, beam_size: int = 3, max_length: int = 128) -> Hypothesis:
        """

        Args:
            token (torch.Tensor): Token tensor (1, sequence_length).
            beam_size (int): Beam size.
            max_length (int): Maximum length.

        Returns:
            torch.Tensor: Output tensor (batch_size, sequence_length, vocab_size).
        """
        assert token.shape[0] == 1, "batch size must be 1"
        init_token = token.squeeze(0).tolist()
        hyps = [Hypothesis(token=init_token, total_score=0.0, caches=[])]
        # NOTE: add a minimum sample to prevent an empty hypothesis
        final_hyps = [Hypothesis(token=init_token, total_score=float("-inf"), caches=[])]
        for _ in range(max_length):
            best_hyps = []
            for hyp in hyps:
                token = torch.tensor([hyp.token], dtype=torch.long, device=token.device)  # (1, sequence_length)
                mask = causal_mask(
                    torch.tensor([len(hyp.token)], device=token.device)
                )  # (1, sequence_length, sequence_length)
                x = self.token_embed(token)
                p = self.position_embed(x)
                x, new_caches = self.decoder.predict(
                    x=x[:, -1:, :] if hyp.caches else x,
                    mask=mask[:, -1:, :] if hyp.caches else mask,
                    caches=hyp.caches,
                    p_q=p[:, -1:, :] if hyp.caches else p,
                    p_k=p,
                )
                x = self.layer_norm(x[:, -1, :])  # (1, d_model)
                x = self.linear(x)  # (1, vocab_size)
                scores = torch.log_softmax(x.squeeze(0), dim=-1)  # (vocab_size,)
                # NOTE: limit the number of candidates to reduce computation
                for score, k in zip(*torch.topk(scores, beam_size)):
                    best_hyps.append(
                        Hypothesis(
                            token=hyp.token + [int(k)],
                            total_score=hyp.total_score + score.item(),
                            caches=new_caches,
                        )
                    )
                best_hyps = sorted(best_hyps, key=lambda x: x.total_score, reverse=True)[:beam_size]
            for best_hyp in best_hyps:
                if best_hyp.token[-1] == self.eos_token_id:
                    final_hyps.append(best_hyp)
            if len(final_hyps) > beam_size:
                break
            hyps = best_hyps
        return max(final_hyps, key=lambda x: x.total_score / len(x.token))

    @classmethod
    def load_pretrained(cls, model_name: str = "Qwen/Qwen3-0.6B") -> "Qwen3":
        src_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="float32", device_map="cpu")
        config = AutoConfig.from_pretrained(model_name)
        model = cls(
            vocab_size=config.vocab_size,
            d_model=config.hidden_size,
            d_k=config.head_dim,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            d_ff=config.intermediate_size,
            num_layers=config.num_hidden_layers,
            dropout_rate=config.attention_dropout,
            max_length=config.max_position_embeddings,
            base=config.rope_parameters["rope_theta"],
            eps=config.rms_norm_eps,
            eos_token_id=config.eos_token_id,
        )
        src_params = sum(v.numel() for v in src_model.state_dict().values())
        num_params = sum(v.numel() for v in model.state_dict().values())
        assert src_params == num_params, f"Number of parameters mismatch: {src_params} != {num_params}"
        new_state_dict = {}
        for key, value in tqdm(src_model.state_dict().items()):
            for pattern in TRANSFORMERS_KEY_MAP.keys():
                match = re.match(pattern, key)
                if match:
                    new_key = TRANSFORMERS_KEY_MAP[pattern]
                    if "{layer_id}" in new_key:
                        layer_id = match.group(1)
                        new_key = new_key.format(layer_id=layer_id)
                        # weight compatibility
                        if "w_q" in new_key:
                            if new_key.endswith("weight"):
                                value = permute_weights(value, config.num_attention_heads)
                            else:
                                value = permute_bias(value, config.num_attention_heads)
                        if "w_k" in new_key:
                            if new_key.endswith("weight"):
                                value = permute_weights(value, config.num_key_value_heads)
                            else:
                                value = permute_bias(value, config.num_key_value_heads)
                        if "n_q" in new_key:
                            value = permute_norm(value)
                        if "n_k" in new_key:
                            value = permute_norm(value)
                    new_state_dict[new_key] = value
                    break
        model.load_state_dict(new_state_dict)
        return model
