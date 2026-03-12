import torch

from slp.modules.rnnt.joint_network import JointNetwork
from slp.modules.rnnt.prediction_network import PredictionNetwork
from slp.modules.rnnt.types import Sequence


class TransducerBeamSearch:
    """Beam search decoder for RNN-T model.

    Proposed in A. Graves, "Sequence transduction with recurrent neural networks,"
    arXiv preprint arXiv:1211.3711, 2012.

    """

    def __init__(
        self,
        prediction_network: PredictionNetwork,
        joint_network: JointNetwork,
        beam_width: int,
        blank_token_id: int,
    ):
        if prediction_network.embed.num_embeddings < 2:
            raise ValueError("vocab_size must be >= 2 (at least blank + one label).")
        self.prediction_network = prediction_network
        self.joint_network = joint_network
        self.beam_width = beam_width
        self.blank_token_id = blank_token_id
        self.B: list[Sequence] = []

    def reset(self) -> None:
        self.B = []

    @torch.inference_mode()
    def search(self, x: torch.Tensor) -> list[Sequence]:
        """Search for the most probable output sequence given encoder output using beam search.

        Args:
            x (torch.Tensor): Encoder output (1, frame_length, encoder_size).

        Returns:
            list[Sequence]: Current beam hypotheses sorted by total_score descending.
        """
        if not self.B:
            h, c = self.prediction_network.init_state(1, x.device)
            self.B = [Sequence(token=[self.blank_token_id], hidden_state=h, cell_state=c, total_score=0.0)]
        for t in range(x.shape[1]):
            A = self.B
            B: list[Sequence] = []

            # NOTE: the prefix search part was removed based on https://arxiv.org/pdf/2201.05420.

            # while B contains less than W elements more probable than the most probable in A
            while (
                len([_y for _y in B if _y.total_score > max(A, key=lambda seq: seq.total_score).total_score])
                < self.beam_width
            ):
                # y^* = most probable in A
                y_star = max(A, key=lambda x: x.total_score)
                # Remove y^* from A
                A = [_y for _y in A if _y is not y_star]
                # Pr(y^*) = Pr(y^*) Pr(∅|y^*, t)
                z, h, c = self.prediction_network(
                    token=torch.tensor([y_star.token[-1:]], dtype=torch.long, device=x.device),
                    hidden_state=y_star.hidden_state,
                    cell_state=y_star.cell_state,
                )
                scores = torch.log_softmax(
                    self.joint_network(x[:, t : t + 1, None, :], z[:, None, :, :]), dim=-1
                ).squeeze()
                base_score = y_star.total_score
                y_star.total_score += scores[self.blank_token_id].item()
                # Add y^* to B
                B.append(y_star)
                # for k in Y: Pr(y^* + k) = Pr(y^*) Pr(k|y^*, t)
                non_blank_scores = scores.clone()
                non_blank_scores[self.blank_token_id] = float("-inf")
                for score, k in zip(*torch.topk(non_blank_scores, min(self.beam_width, scores.shape[0] - 1))):
                    A.append(
                        Sequence(
                            token=y_star.token + [k.item()],
                            hidden_state=h,
                            cell_state=c,
                            total_score=base_score + score.item(),
                        )
                    )
            # Remove all but the W most probable from B
            self.B = sorted(B, key=lambda x: x.total_score, reverse=True)[: self.beam_width]
        return self.B

    def best_token_ids(self) -> list[int]:
        """Return y with highest log Pr(y)/|y| in B"""
        if not self.B:
            return []
        best = max(self.B, key=lambda s: s.total_score / len(s.token))
        return best.token[1:]  # Remove the initial blank token
