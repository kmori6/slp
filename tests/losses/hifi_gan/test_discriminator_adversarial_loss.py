import torch

from slp.losses.hifi_gan.discriminator_adversarial_loss import DiscriminatorAdversarialLoss


def test_backward_gradients():
    loss_fn = DiscriminatorAdversarialLoss()
    real_xs = [torch.randn(3, 16, requires_grad=True), torch.randn(3, 8, requires_grad=True)]
    fake_xs = [torch.randn(3, 16, requires_grad=True), torch.randn(3, 8, requires_grad=True)]

    loss = loss_fn(real_xs, fake_xs)
    loss.backward()

    for real_x, fake_x in zip(real_xs, fake_xs):
        assert real_x.grad is not None and fake_x.grad is not None
