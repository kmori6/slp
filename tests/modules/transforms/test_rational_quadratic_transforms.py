import torch

from slp.modules.transforms.rational_quadratic_transforms import rational_quadratic_transforms


def test_forward_inverse():
    torch.manual_seed(0)
    B = 5.0
    K = 10
    inputs = torch.randn(4, 64, 100)
    unnorm_w = torch.randn(4, 64, 100, K)
    unnorm_h = torch.randn(4, 64, 100, K)
    unnorm_d = torch.randn(4, 64, 100, K - 1)

    outputs, log_det_fwd = rational_quadratic_transforms(inputs, unnorm_w, unnorm_h, unnorm_d, B=B)
    inverted, log_det_inv = rational_quadratic_transforms(outputs, unnorm_w, unnorm_h, unnorm_d, inverse=True, B=B)

    torch.testing.assert_close(inverted, inputs, atol=1e-3, rtol=1e-4)
    torch.testing.assert_close(log_det_fwd, -log_det_inv, atol=1e-4, rtol=1e-4)
