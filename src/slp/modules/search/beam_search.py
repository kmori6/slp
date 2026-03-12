from typing import Protocol

import torch

from slp.modules.transformer.types import Cache, Hypothesis


class Decodable(Protocol):
    def embed(self, token_ids: torch.Tensor) -> torch.Tensor: ...

    def logits(self, x: torch.Tensor) -> torch.Tensor: ...

    def predict(
        self,
        x_enc: torch.Tensor,
        x_dec: torch.Tensor,
        caches: list[Cache],
    ) -> tuple[torch.Tensor, list[Cache]]: ...


class BeamSearch:
    def __init__(self, decoder: Decodable, bos_token_id: int, eos_token_id: int):
        self.decoder = decoder
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    @torch.inference_mode()
    def search(
        self,
        x_enc: torch.Tensor,
        beam_size: int,
        length_buffer: int,
        length_penalty: float,
    ) -> Hypothesis:
        """

        Args:
            x_enc (torch.Tensor): Encoder output of shape (1, src_len, d_model).
            beam_size (int): Maximum number of active hypotheses per step.
            length_buffer (int): Extra decoding steps added to ``src_len``.
            length_penalty (float): Exponent used for length-normalized scoring.

        Returns:
            Hypothesis: Best hypothesis after beam search, selected by
                ``total_score / (len(token) ** length_penalty)``.
        """
        device = x_enc.device
        if beam_size < 1:
            raise ValueError(f"beam_size must be >= 1, but got {beam_size}.")
        hyps = [Hypothesis(token=[self.bos_token_id], total_score=0.0, caches=[])]
        final_hyps: list[Hypothesis] = []

        max_steps = x_enc.shape[1] + length_buffer
        for _ in range(max_steps):
            candidates = []
            for hyp in hyps:
                last_token = torch.tensor([[hyp.token[-1]]], dtype=torch.long, device=device)
                x_dec = self.decoder.embed(last_token)  # (1, 1, d_model)
                x_dec, new_caches = self.decoder.predict(x_enc, x_dec, hyp.caches)
                logits = self.decoder.logits(x_dec)  # (1, 1, vocab_size)
                scores = torch.log_softmax(logits.squeeze(0).squeeze(0), dim=-1)  # (vocab_size,)
                # NOTE: limit candidates to beam_size per hypothesis
                top_k = min(beam_size, scores.shape[0])
                top_scores, top_ids = torch.topk(scores, top_k)
                for score, token_id in zip(top_scores, top_ids):
                    candidates.append(
                        Hypothesis(
                            token=hyp.token + [int(token_id)],
                            total_score=hyp.total_score + score.item(),
                            caches=new_caches,
                        )
                    )

            candidates = sorted(candidates, key=lambda h: h.total_score, reverse=True)
            hyps = []
            for c in candidates[:beam_size]:
                if c.token[-1] == self.eos_token_id:
                    final_hyps.append(c)
                else:
                    hyps.append(c)

            if len(final_hyps) >= beam_size or not hyps:
                break

        result_hyps = final_hyps if final_hyps else hyps
        return max(result_hyps, key=lambda h: h.total_score / (len(h.token) ** length_penalty))
