import torch
import torch.nn as nn


class FeatureMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss_fn = nn.L1Loss(reduction="mean")

    def forward(self, real_feats: list[list[torch.Tensor]], fake_feats: list[list[torch.Tensor]]) -> torch.Tensor:
        """

        L_FM(G; D) = sum_{k=1}^K E_(x,s)[sum_{t=1}^T 1/N_i |D^i_t(x) - D^i_t(G(s))|]

        Args:
            real_feats (list[torch.Tensor]): Discriminator feature tensors from the ground truth audio (batch, *).
            fake_feats (list[torch.Tensor]): Discriminator feature tensors from the generator (batch, *).

        Returns:
            torch.Tensor: Feature matching loss.
        """
        if len(real_feats) != len(fake_feats):
            raise ValueError(
                f"Length mismatch between real_feats and fake_feats: {len(real_feats)} vs {len(fake_feats)}"
            )
        losses = []
        for real_disc_feats, fake_disc_feats in zip(real_feats, fake_feats):
            for real_layer_feats, fake_layer_feats in zip(real_disc_feats, fake_disc_feats):
                if real_layer_feats.shape != fake_layer_feats.shape:
                    raise ValueError(f"Shape mismatch: {real_layer_feats.shape} vs {fake_layer_feats.shape}")
                # NOTE: L1 loss averaged over the batch and feature size
                loss = self.l1_loss_fn(real_layer_feats, fake_layer_feats)
                losses.append(loss)
        loss = torch.stack(losses).sum()
        return loss
