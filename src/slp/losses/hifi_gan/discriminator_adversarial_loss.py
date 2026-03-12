import torch
import torch.nn as nn


class DiscriminatorAdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss_fn = nn.MSELoss(reduction="mean")

    def forward(self, real_xs: list[torch.Tensor], fake_xs: list[torch.Tensor]) -> torch.Tensor:
        """

        Args:
            real_xs (list[torch.Tensor]): Discriminator output tensors from the ground truth audio (batch, *).
            fake_xs (list[torch.Tensor]): Discriminator output tensors from the generator (batch, *).

        Returns:
            torch.Tensor: Discriminator adversarial loss.
        """
        losses = []
        for real_x, fake_x in zip(real_xs, fake_xs):
            if real_x.shape != fake_x.shape:
                raise ValueError(f"Shape mismatch between real_x and fake_x: {real_x.shape} vs {fake_x.shape}")
            loss = self.mse_loss_fn(real_x, torch.ones_like(real_x)) + self.mse_loss_fn(
                fake_x, torch.zeros_like(fake_x)
            )
            losses.append(loss)
        loss = torch.stack(losses).sum()
        return loss
