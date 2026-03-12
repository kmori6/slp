import torch

from slp.losses.hifi_gan.feature_matching_loss import FeatureMatchingLoss


def test_backward_gradients():
    loss_fn = FeatureMatchingLoss()
    real_feats = [
        [torch.randn(2, 16, 20, requires_grad=True), torch.randn(2, 32, 10, requires_grad=True)],
        [torch.randn(2, 16, 12, requires_grad=True)],
    ]
    fake_feats = [
        [torch.randn(2, 16, 20, requires_grad=True), torch.randn(2, 32, 10, requires_grad=True)],
        [torch.randn(2, 16, 12, requires_grad=True)],
    ]

    loss = loss_fn(real_feats, fake_feats)
    loss.backward()

    for disc_feats in real_feats + fake_feats:
        for layer_feats in disc_feats:
            assert layer_feats.grad is not None
