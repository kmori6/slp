import torch
import torch.nn as nn


class GeneratorAdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss_fn = nn.MSELoss(reduction="mean")

    def forward(self, fake_xs: list[torch.Tensor]) -> torch.Tensor:
        """

        L_adv = sum_{k=1}^K E_s[(D_k(G(s)) - 1)^2]
            - G: generator
            - D_k: k-th discriminator
            - K: number of discriminators
            - s: input condition

        Args:
            fake_xs (list[torch.Tensor]): Discriminator output tensors from the generator (batch, *).

        Returns:
            torch.Tensor: Generator adversarial loss.
        """
        losses = []
        for fake_x in fake_xs:
            loss = self.mse_loss_fn(fake_x, torch.ones_like(fake_x))
            losses.append(loss)
        loss = torch.stack(losses).sum()
        return loss
