import torch
import torch.nn.functional as F


def make_knots(
    theta_w: torch.Tensor,
    theta_h: torch.Tensor,
    B: float,
    min_bin_width: float = 1e-3,
    min_bin_height: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """

    Args:
        theta_w (torch.Tensor): Unnormalized widths (..., K).
        theta_h (torch.Tensor): Unnormalized heights (..., K).
        B (float): Boundary for the spline region [-B, B].
        min_bin_width (float): Minimum width for numerical stability.
        min_bin_height (float): Minimum height for numerical stability.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            xk (torch.Tensor): Knot x-coordinates (..., K+1).
            yk (torch.Tensor): Knot y-coordinates (..., K+1).
    """
    K = theta_w.shape[-1]

    # widths and heights of the K bins
    w = F.softmax(theta_w, dim=-1)
    h = F.softmax(theta_h, dim=-1)

    # w_i >= min_bin_width, h_i >= min_bin_height for numerical stability
    # reference: https://github.com/bayesiains/nsf/blob/master/nde/transforms/splines/quadratic.py
    w = min_bin_width + (1 - K * min_bin_width) * w
    h = min_bin_height + (1 - K * min_bin_height) * h

    w = torch.cumsum(w * (2 * B), dim=-1)
    h = torch.cumsum(h * (2 * B), dim=-1)

    x_ks = F.pad(w, (1, 0), value=0.0) - B  # (..., K + 1)
    y_ks = F.pad(h, (1, 0), value=0.0) - B  # (..., K + 1)

    # (x^(0), y^(0)) = (−B, −B)
    x_ks[..., 0] = y_ks[..., 0] = -B

    # (x^(K), y^(K)) = (B, B)
    x_ks[..., -1] = y_ks[..., -1] = B

    return x_ks, y_ks


def make_derivatives(theta_d: torch.Tensor, min_derivative: float = 1e-3):
    """

    Args:
        theta_d (torch.Tensor): Unnormalized derivative (..., K-1)
        min_derivatives (float): Minimum derivative for numerical stability.

    Return:
        torch.Tensor: Derivatives (..., K+1).
    """
    # d_i >= min_derivative
    # reference: https://github.com/bayesiains/nsf/blob/master/nde/transforms/splines/quadratic.py
    d = min_derivative + F.softplus(theta_d)  # (..., K - 1)

    # delta^(0) = delta^(K) = 1
    d = F.pad(d, (1, 1), value=1.0)  # (..., K + 1)
    return d


def rational_quadratic_transforms(
    inputs: torch.Tensor,
    theta_w: torch.Tensor,
    theta_h: torch.Tensor,
    theta_d: torch.Tensor,
    inverse: bool = False,
    B: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Monotonic rational-quadratic transforms.

    Args:
        inputs (torch.Tensor): Input values (...).
        theta_w (torch.Tensor): Raw width parameters (..., K).
        theta_h (torch.Tensor): Raw height parameters (..., K).
        theta_d (torch.Tensor): Raw derivative parameters (..., K-1).
        inverse (bool): If True, compute the inverse transform.
        B (float): Boundary B for the spline region [-B, B].

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            outputs (torch.Tensor): Transformed values (...).
            logabsdet (torch.Tensor): Log absolute determinant per element (...).
    """

    x_ks, y_ks = make_knots(theta_w, theta_h, B=B)
    d = make_derivatives(theta_d)

    inside = (-B <= inputs) & (inputs <= B)

    # tails -> identity
    transforms = inputs.clone()
    log_abs_det = torch.zeros_like(inputs)

    if inside.any():
        # flatten for simpler gather
        inputs_flat = inputs[inside]  # (*,)
        x_ks_flat = x_ks[inside]  # (*, K + 1)
        y_ks_flat = y_ks[inside]  # (*, K + 1)
        d_flat = d[inside]  # (*, K + 1)

        if inverse:
            # for y \in [y^(k), y^(k+1)]
            k = (y_ks_flat < inputs_flat[..., None]).sum(dim=-1) - 1
        else:
            # for x \in [x^(k), x^(k+1)]
            k = (x_ks_flat < inputs_flat[:, None]).sum(dim=-1) - 1  # (*,)

        x_k = x_ks_flat.gather(-1, k[:, None]).squeeze(-1)  # (*,)
        x_k_plus_1 = x_ks_flat.gather(-1, (k + 1)[:, None]).squeeze(-1)  # (*,)
        y_k = y_ks_flat.gather(-1, k[:, None]).squeeze(-1)  # (*,)
        y_k_plus_1 = y_ks_flat.gather(-1, (k + 1)[:, None]).squeeze(-1)  # (*,)
        d_k = d_flat.gather(-1, k[:, None]).squeeze(-1)  # (*,)
        d_k_plus_1 = d_flat.gather(-1, (k + 1)[:, None]).squeeze(-1)  # (*,)

        # Appendix A.1: Parameterization of the spline
        w_k = x_k_plus_1 - x_k
        h_k = y_k_plus_1 - y_k
        s_k = h_k / w_k

        if inverse:
            # Appendix A.3: Computing the inverse
            yd = inputs_flat - y_k

            a = h_k * (s_k - d_k) + yd * (d_k_plus_1 + d_k - 2 * s_k)
            b = h_k * d_k - yd * (d_k_plus_1 + d_k - 2 * s_k)
            c = -s_k * yd

            xi = 2 * c / (-b - torch.sqrt((b.pow(2) - 4 * a * c).clamp(min=0.0)))  # (29)

            # xi = (x - x^(k)) / w^(k) -> x = x^(k) + xi * w^(k)
            transforms[inside] = x_k + xi * w_k  # x = g^(-1)(y)

            # Appendix A.2: Coputing the derivative
            denom = s_k + (d_k_plus_1 + d_k - 2 * s_k) * xi * (1 - xi)
            dy_dx = (
                s_k.pow(2) * (d_k_plus_1 * xi.pow(2) + 2 * s_k * xi * (1 - xi) + d_k * (1 - xi).pow(2)) / denom.pow(2)
            )
            # log|dx/dy| = -log|dy/dx|
            log_abs_det[inside] = -torch.log(dy_dx)

        else:
            xi = (inputs_flat - x_k) / w_k  # xi \in [0, 1]

            numer = h_k * (s_k * xi.pow(2) + d_k * xi * (1 - xi))
            denom = s_k + (d_k_plus_1 + d_k - 2 * s_k) * xi * (1 - xi)

            transforms[inside] = y_k + numer / denom  # y = g(x) (19)

            # Appendix A.2: Coputing the derivative
            dy_dx = (
                s_k.pow(2) * (d_k_plus_1 * xi.pow(2) + 2 * s_k * xi * (1 - xi) + d_k * (1 - xi).pow(2)) / denom.pow(2)
            )
            log_abs_det[inside] = torch.log(dy_dx)

    return transforms, log_abs_det
