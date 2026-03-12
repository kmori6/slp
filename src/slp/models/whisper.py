import re

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import WhisperConfig, WhisperForConditionalGeneration

from slp.modules.transformer.decoder import Decoder
from slp.modules.transformer.encoder import Encoder
from slp.modules.transformer.feed_forward import FeedForward
from slp.modules.transformer.multi_head_attention import MultiHeadAttention
from slp.modules.transformer.positional_encoding import sinusoidal_positional_encoding
from slp.modules.transformer.pre_norm_decoder_layer import PreNormDecoderLayer
from slp.modules.transformer.pre_norm_encoder_layer import PreNormEncoderLayer
from slp.modules.transformer.types import Hypothesis
from slp.utils.mask import causal_mask, sequence_mask

TRANSFORMERS_KEY_MAP = {
    r"model\.encoder\.conv1\.weight": "subsampling.conv1.weight",
    r"model\.encoder\.conv1\.bias": "subsampling.conv1.bias",
    r"model\.encoder\.conv2\.weight": "subsampling.conv2.weight",
    r"model\.encoder\.conv2\.bias": "subsampling.conv2.bias",
    r"model\.encoder\.embed_positions\.weight": "src_position_embedding",
    r"model\.encoder\.layers\.(\d+)\.self_attn\.q_proj\.weight": "encoder.layers.{layer_id}.mha.w_q.weight",
    r"model\.encoder\.layers\.(\d+)\.self_attn\.q_proj\.bias": "encoder.layers.{layer_id}.mha.w_q.bias",
    r"model\.encoder\.layers\.(\d+)\.self_attn\.k_proj\.weight": "encoder.layers.{layer_id}.mha.w_k.weight",
    r"model\.encoder\.layers\.(\d+)\.self_attn\.v_proj\.weight": "encoder.layers.{layer_id}.mha.w_v.weight",
    r"model\.encoder\.layers\.(\d+)\.self_attn\.v_proj\.bias": "encoder.layers.{layer_id}.mha.w_v.bias",
    r"model\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.weight": "encoder.layers.{layer_id}.mha.w_o.weight",
    r"model\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.bias": "encoder.layers.{layer_id}.mha.w_o.bias",
    r"model\.encoder\.layers\.(\d+)\.fc1\.weight": "encoder.layers.{layer_id}.ffn.w_1.weight",
    r"model\.encoder\.layers\.(\d+)\.fc1\.bias": "encoder.layers.{layer_id}.ffn.w_1.bias",
    r"model\.encoder\.layers\.(\d+)\.fc2\.weight": "encoder.layers.{layer_id}.ffn.w_2.weight",
    r"model\.encoder\.layers\.(\d+)\.fc2\.bias": "encoder.layers.{layer_id}.ffn.w_2.bias",
    r"model\.encoder\.layers\.(\d+)\.self_attn_layer_norm\.weight": "encoder.layers.{layer_id}.layer_norm1.weight",
    r"model\.encoder\.layers\.(\d+)\.self_attn_layer_norm\.bias": "encoder.layers.{layer_id}.layer_norm1.bias",
    r"model\.encoder\.layers\.(\d+)\.final_layer_norm\.weight": "encoder.layers.{layer_id}.layer_norm2.weight",
    r"model\.encoder\.layers\.(\d+)\.final_layer_norm\.bias": "encoder.layers.{layer_id}.layer_norm2.bias",
    r"model\.encoder\.layer_norm\.weight": "layer_norm1.weight",
    r"model\.encoder\.layer_norm\.bias": "layer_norm1.bias",
    r"model\.decoder\.embed_tokens\.weight": "token_embedding.weight",
    r"model\.decoder\.embed_positions\.weight": "tgt_position_embedding.weight",
    r"model\.decoder\.layers\.(\d+)\.self_attn\.q_proj\.weight": "decoder.layers.{layer_id}.masked_mha.w_q.weight",
    r"model\.decoder\.layers\.(\d+)\.self_attn\.q_proj\.bias": "decoder.layers.{layer_id}.masked_mha.w_q.bias",
    r"model\.decoder\.layers\.(\d+)\.self_attn\.k_proj\.weight": "decoder.layers.{layer_id}.masked_mha.w_k.weight",
    r"model\.decoder\.layers\.(\d+)\.self_attn\.v_proj\.weight": "decoder.layers.{layer_id}.masked_mha.w_v.weight",
    r"model\.decoder\.layers\.(\d+)\.self_attn\.v_proj\.bias": "decoder.layers.{layer_id}.masked_mha.w_v.bias",
    r"model\.decoder\.layers\.(\d+)\.self_attn\.out_proj\.weight": "decoder.layers.{layer_id}.masked_mha.w_o.weight",
    r"model\.decoder\.layers\.(\d+)\.self_attn\.out_proj\.bias": "decoder.layers.{layer_id}.masked_mha.w_o.bias",
    r"model\.decoder\.layers\.(\d+)\.encoder_attn\.q_proj\.weight": "decoder.layers.{layer_id}.mha.w_q.weight",
    r"model\.decoder\.layers\.(\d+)\.encoder_attn\.q_proj\.bias": "decoder.layers.{layer_id}.mha.w_q.bias",
    r"model\.decoder\.layers\.(\d+)\.encoder_attn\.k_proj\.weight": "decoder.layers.{layer_id}.mha.w_k.weight",
    r"model\.decoder\.layers\.(\d+)\.encoder_attn\.v_proj\.weight": "decoder.layers.{layer_id}.mha.w_v.weight",
    r"model\.decoder\.layers\.(\d+)\.encoder_attn\.v_proj\.bias": "decoder.layers.{layer_id}.mha.w_v.bias",
    r"model\.decoder\.layers\.(\d+)\.encoder_attn\.out_proj\.weight": "decoder.layers.{layer_id}.mha.w_o.weight",
    r"model\.decoder\.layers\.(\d+)\.encoder_attn\.out_proj\.bias": "decoder.layers.{layer_id}.mha.w_o.bias",
    r"model\.decoder\.layers\.(\d+)\.fc1\.weight": "decoder.layers.{layer_id}.ffn.w_1.weight",
    r"model\.decoder\.layers\.(\d+)\.fc1\.bias": "decoder.layers.{layer_id}.ffn.w_1.bias",
    r"model\.decoder\.layers\.(\d+)\.fc2\.weight": "decoder.layers.{layer_id}.ffn.w_2.weight",
    r"model\.decoder\.layers\.(\d+)\.fc2\.bias": "decoder.layers.{layer_id}.ffn.w_2.bias",
    r"model\.decoder\.layers\.(\d+)\.self_attn_layer_norm\.weight": "decoder.layers.{layer_id}.layer_norm1.weight",
    r"model\.decoder\.layers\.(\d+)\.self_attn_layer_norm\.bias": "decoder.layers.{layer_id}.layer_norm1.bias",
    r"model\.decoder\.layers\.(\d+)\.encoder_attn_layer_norm\.weight": "decoder.layers.{layer_id}.layer_norm2.weight",
    r"model\.decoder\.layers\.(\d+)\.encoder_attn_layer_norm\.bias": "decoder.layers.{layer_id}.layer_norm2.bias",
    r"model\.decoder\.layers\.(\d+)\.final_layer_norm\.weight": "decoder.layers.{layer_id}.layer_norm3.weight",
    r"model\.decoder\.layers\.(\d+)\.final_layer_norm\.bias": "decoder.layers.{layer_id}.layer_norm3.bias",
    r"model\.decoder\.layer_norm\.weight": "layer_norm2.weight",
    r"model\.decoder\.layer_norm\.bias": "layer_norm2.bias",
    r"proj_out\.weight": "linear.weight",
}


class ConvolutionSubsampling(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, output_size, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(output_size, output_size, kernel_size=3, stride=2, padding=1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            x (torch.Tensor): Embedding tensor (batch_size, frame_length, input_size).
            mask (torch.Tensor): Mask tensor (batch_size, frame_length).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                torch.Tensor: Output tensor (batch_size, (frame_size + 1) // 2, output_size).
                torch.Tensor: Mask tensor (batch_size, (frame_size + 1) // 2).
        """
        x = x.transpose(1, 2)  # (batch_size, input_size, frame_length)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)  # (batch_size, input_size, (frame_length + 1) // 2)
        x = self.activation(x)
        return x.transpose(1, 2), sequence_mask((mask.sum(-1) + 1) // 2)


class Whisper(nn.Module):
    def __init__(
        self,
        mel_size: int,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_enc_layers: int,
        num_dec_layers: int,
        dropout_rate: float,
        pad_token_id: int,
        eos_token_id: int,
        src_position_length: int,
        tgt_position_length: int,
    ):
        super().__init__()
        self.mel_size = mel_size
        self.eos_token_id = eos_token_id
        self.tgt_position_length = tgt_position_length
        self.subsampling = ConvolutionSubsampling(mel_size, d_model)
        self.register_buffer("src_position_embedding", sinusoidal_positional_encoding(d_model, src_position_length))
        self.dropout1 = nn.Dropout(dropout_rate)
        self.encoder = Encoder(
            encoder_layer=PreNormEncoderLayer(
                mha_module=MultiHeadAttention(
                    d_model,
                    d_k=d_model // num_heads,
                    num_heads=num_heads,
                    num_kv_heads=num_heads,
                    dropout_rate=dropout_rate,
                    query_bias=True,
                    value_bias=True,
                    output_bias=True,
                ),
                mha_norm_module=nn.LayerNorm(d_model),
                ffn_module=FeedForward(d_model, d_ff, dropout_rate, activation=nn.GELU()),
                ffn_norm_module=nn.LayerNorm(d_model),
                dropout_rate=dropout_rate,
            ),
            num_layers=num_enc_layers,
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.tgt_position_embedding = nn.Embedding(tgt_position_length, d_model)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.decoder = Decoder(
            decoder_layer=PreNormDecoderLayer(
                masked_mha_module=MultiHeadAttention(
                    d_model,
                    d_k=d_model // num_heads,
                    num_heads=num_heads,
                    num_kv_heads=num_heads,
                    dropout_rate=dropout_rate,
                    query_bias=True,
                    value_bias=True,
                    output_bias=True,
                ),
                masked_mha_norm_module=nn.LayerNorm(d_model),
                mha_module=MultiHeadAttention(
                    d_model,
                    d_k=d_model // num_heads,
                    num_heads=num_heads,
                    num_kv_heads=num_heads,
                    dropout_rate=dropout_rate,
                    query_bias=True,
                    value_bias=True,
                    output_bias=True,
                ),
                mha_norm_module=nn.LayerNorm(d_model),
                ffn_module=FeedForward(d_model, d_ff, dropout_rate, activation=nn.GELU()),
                ffn_norm_module=nn.LayerNorm(d_model),
                dropout_rate=dropout_rate,
            ),
            num_layers=num_dec_layers,
        )
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size, bias=False)

        # tie weights
        self.linear.weight = self.token_embedding.weight

    def forward(
        self,
        input_features: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """

        Args:
            input_features (torch.Tensor): Acoustic features
                (batch_size, frame_length, mel_size).
            decoder_input_ids (torch.Tensor): Decoder input token sequence (batch_size, sequence_length).
            attention_mask (torch.Tensor): Encoder sequence mask (batch_size, frame_length).
            decoder_attention_mask (torch.Tensor | None): Decoder sequence mask (batch_size, sequence_length).

        Returns:
            torch.Tensor: Output tensor (batch_size, sequence_length, vocab_size).
        """
        # encoder
        x_enc, mask_enc = self.encode(input_features, attention_mask)  # (batch_size, frame_length, d_model)
        # decoder
        if decoder_attention_mask is None:
            token_length = torch.full(
                (decoder_input_ids.shape[0],),
                decoder_input_ids.shape[1],
                dtype=torch.long,
                device=decoder_input_ids.device,
            )
        else:
            token_length = decoder_attention_mask.sum(-1).to(torch.long)
        mask_dec = causal_mask(token_length)  # (batch_size, sequence_length, sequence_length)
        x_dec = self.token_embedding(decoder_input_ids)  # (batch_size, sequence_length, d_model)
        x_dec = x_dec + self.tgt_position_embedding(torch.arange(x_dec.shape[1], device=x_enc.device)[None, :])
        x_dec = self.dropout2(x_dec)
        x_dec = self.decoder(x_enc, x_dec, mask_enc[:, None, :].expand(-1, x_dec.shape[1], -1), mask_dec)
        x_dec = self.layer_norm2(x_dec)
        x_dec = self.linear(x_dec)  # (batch_size, sequence_length, vocab_size)
        return x_dec

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            x (torch.Tensor): Acoustic embedding tensor (batch_size, frame_length, input_size).
            mask (torch.Tensor): Encoder sequence mask (batch_size, frame_length).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                torch.Tensor: Encoder output tensor (batch_size, frame_length, d_model).
                torch.Tensor: Encoder mask tensor (batch_size, frame_length).
        """
        x, mask = self.subsampling(x, mask)  # (batch_size, frame_length, d_model)
        x = x + self.src_position_embedding[None, : x.shape[1], :].to(x.device)  # type: ignore[index]
        x = self.dropout1(x)
        x = self.encoder(x, mask[:, None, :].expand(-1, x.shape[1], -1))
        x = self.layer_norm1(x)
        return x, mask

    @torch.inference_mode()
    def predict(
        self,
        input_features: torch.Tensor,
        prompt_token: list[int],
        attention_mask: torch.Tensor,
        beam_size: int = 3,
    ) -> Hypothesis:
        """

        Args:
            input_features (torch.Tensor): Acoustic features
                (1, frame_length, mel_size).
            prompt_token (list[int]): Prompt token sequence.
                currently, only support notimestamp mode '<|en|><|translate|><|notimestamps|>'.
            attention_mask (torch.Tensor): Encoder sequence mask (1, frame_length).

        """
        x_enc, _ = self.encode(input_features, attention_mask)  # (1, frame_length, d_model)
        # decode
        hyps = [Hypothesis(token=prompt_token, total_score=0.0, caches=[])]
        # NOTE: add a minimum sample to prevent an empty hypothesis
        final_hyps = [Hypothesis(token=prompt_token, total_score=float("-inf"), caches=[])]
        for _ in range(self.tgt_position_length - len(prompt_token)):
            best_hyps = []
            for hyp in hyps:
                token = torch.tensor([hyp.token], dtype=torch.long, device=x_enc.device)  # (1, sequence_length)
                x = self.token_embedding(token)  # (1, sequence_length, d_model)
                x = x + self.tgt_position_embedding(torch.arange(len(hyp.token), device=x_enc.device)[None, :])
                x, new_caches = self.decoder.predict(
                    x_enc,
                    x[:, -1:, :] if hyp.caches else x,
                    hyp.caches,
                )
                x = self.layer_norm2(x[:, -1, :])  # (1, d_model)
                x = self.linear(x)  # (1, vocab_size)
                scores = x.squeeze(0).log_softmax(dim=-1)  # (vocab_size,)
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
    def load_pretrained(cls, model_name: str = "openai/whisper-large-v3-turbo") -> "Whisper":
        src_model = WhisperForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="float32", low_cpu_mem_usage=True, use_safetensors=True, device_map="cpu"
        )
        config = WhisperConfig.from_pretrained(model_name)
        model = cls(
            mel_size=config.num_mel_bins,
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            num_heads=config.encoder_attention_heads,  # encoder_attention_heads is the same as decoder_attention_heads
            d_ff=config.encoder_ffn_dim,  # encoder_ffn_dim is the same as decoder_ffn_dim
            num_enc_layers=config.encoder_layers,
            num_dec_layers=config.decoder_layers,
            dropout_rate=config.dropout,
            pad_token_id=config.pad_token_id,  # type: ignore[arg-type]
            eos_token_id=config.eos_token_id,  # type: ignore[arg-type]
            src_position_length=config.max_source_positions,
            tgt_position_length=config.max_target_positions,
        )
        new_state_dict = {}
        for key, value in tqdm(src_model.state_dict().items()):
            for pattern in TRANSFORMERS_KEY_MAP.keys():
                match = re.match(pattern, key)
                if match:
                    new_key = TRANSFORMERS_KEY_MAP[pattern]
                    if "{layer_id}" in new_key:
                        layer_id = match.group(1)
                        new_key = new_key.format(layer_id=layer_id)
                    new_state_dict[new_key] = value
                    break
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        assert missing_keys == [], f"Missing keys: {missing_keys}"
        assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"
        return model
