import torch

from slp.losses.hifi_gan.generator_adversarial_loss import GeneratorAdversarialLoss


def test_backward_gradients():
    loss_fn = GeneratorAdversarialLoss()
    fake_xs = [torch.randn(3, 16, requires_grad=True), torch.randn(3, 8, requires_grad=True)]

    loss = loss_fn(fake_xs)
    loss.backward()

    for fake_x in fake_xs:
        assert fake_x.grad is not None
