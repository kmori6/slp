import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from slp.modules.vits.condition_encoder import ConditionEncoder
from slp.modules.vits.spline_coupling_layer import SplineCouplingLayer


class StochasticDurationPredictor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
        dropout_rate: float,
        cond_size: int,
        num_flow_blocks: int,
        num_prior_layers: int,
        num_condition_layers: int,
        num_posterior_layers: int,
        spline_num_bins: int,
        spline_bound: float,
    ):
        super().__init__()

        # flow
        self.prior_shift = nn.Parameter(torch.zeros(2, 1))
        self.prior_log_scale = nn.Parameter(torch.zeros(2, 1))
        self.prior_layers = nn.ModuleList(
            [
                SplineCouplingLayer(
                    2,
                    hidden_size,
                    kernel_size,
                    num_blocks=num_flow_blocks,
                    num_bins=spline_num_bins,
                    B=spline_bound,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_prior_layers)
            ]
        )

        self.duration_condition_encoder = ConditionEncoder(
            input_size=1,
            hidden_size=hidden_size,
            kernel_size=kernel_size,
            num_layers=num_condition_layers,
            dropout_rate=dropout_rate,
        )

        # posterior encoder
        self.posterior_shift = nn.Parameter(torch.zeros(2, 1))
        self.posterior_log_scale = nn.Parameter(torch.zeros(2, 1))
        self.posterior_layers = nn.ModuleList(
            [
                SplineCouplingLayer(
                    2,
                    hidden_size,
                    kernel_size,
                    num_blocks=num_flow_blocks,
                    num_bins=spline_num_bins,
                    B=spline_bound,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_posterior_layers)
            ]
        )

        self.text_condition_encoder = ConditionEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            kernel_size=kernel_size,
            num_layers=num_condition_layers,
            dropout_rate=dropout_rate,
            cond_size=cond_size,
        )

    def _posterior_forward(
        self, x: torch.Tensor, noise: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Args:
            x (torch.Tensor): Input tensor (batch_size, hidden_size, seq_length).
            noise (torch.Tensor): Gaussian random tensor (batch_size, 2, seq_length).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                torch.Tensor: Random variable `u` (batch_size, 1, seq_length).
                torch.Tensor: Random variable `v` (batch_size, 1, seq_length).
                torch.Tensor: Log posterior distribution `log q(u, v | d, x)` (batch_size,).
        """
        # log q(u, v | d, x) = log N(e; 0, I) - log|det d(u, v)/de|
        log_eps = torch.sum((-0.5 * (math.log(2 * math.pi) + noise.pow(2))), dim=[1, 2])

        b, _, t = x.shape

        # z = exp(s) * e + b
        # log q(u, v | d, x) = log N(e; 0, I) - log|det d(u, v)/dz dz/de|
        # dz/de = diag(exp(s)) -> log|det dz/de| = sum log exp(s) = sum s
        z = noise * torch.exp(self.posterior_log_scale) + self.posterior_shift
        log_det_q = torch.sum(self.posterior_log_scale.expand(b, -1, t), dim=[1, 2])

        # z_k = f(z_k-1) (k=1...n)
        # log q(u, v | d, x) = log N(e; 0, I) - log|det d(u, v)/dz_n ... dz/de|
        for flow in self.posterior_layers:
            z, log_det = flow(z, g=x, reverse=False)
            z = torch.flip(z, dims=[1])
            log_det_q = log_det_q + log_det

        # restrict the support of u to be [0, 1)
        # log q(u, v | d, x) = log N(e; 0, I) - log|det d(u, v)/d(z_u, v) d(z_u, v)/dz_n... dz/de|
        # log|det d(z_u, v)/dz_n| = 0
        # du/dz_u = log sigma(z_u) + log sigma(-z_u)
        # -> log|det d(u, v)/d(z_u, v)| = sum log sigma(z_u) + log sigma(-z_u)
        v, z_u = torch.split(z, [1, 1], dim=1)
        u = torch.sigmoid(z_u)
        log_det_sigmoid = torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)), dim=[1, 2])
        log_det_q = log_det_q + log_det_sigmoid

        log_q = log_eps - log_det_q
        return u, v, log_q

    def _prior_forward(
        self, c_text: torch.Tensor, z: torch.Tensor | None = None, noise_scale: float = 1.0, reverse: bool = False
    ) -> torch.Tensor:
        """

        Args:
            c_text (torch.Tensor): Input tensor (batch_size, hidden_size, seq_length).
            z (torch.Tensor | None): Gaussian random tensor (batch_size, 2, seq_length).
            noise_scale (float): Sampling noise scale.
            reverse (bool): If True, sample log(d-u) else return lower bound.

        Returns:
            torch.Tensor: Log(d-u) (batch_size, 1, seq_length) if reverse is True else lower bound (batch_size,).
        """
        b, _, t = c_text.shape

        if reverse:
            x = torch.randn(b, 2, t, device=c_text.device, dtype=c_text.dtype) * noise_scale

            for flow in reversed(self.prior_layers[1:]):
                x = torch.flip(x, dims=[1])
                x, _ = flow(x, g=c_text, reverse=True)
            x = torch.flip(x, [1])

            x = torch.exp(-self.prior_log_scale) * (x - self.prior_shift)

            # x = [log(d-u), v]
            r, _ = torch.split(x, [1, 1], dim=1)

            return r

        if z is None:
            raise ValueError("z must be provided when reverse=False.")

        # r = log(d-u)
        # log p(d-u, v | c_text) = log p(r, v | c_text) + log|det dr/d(d-u)|
        # dr/d(d-u) = 1/r -> log|det dr/d(d-u)| = sum (-r)
        r, _ = torch.split(z, [1, 1], dim=1)
        log_det_p = torch.sum(-r, dim=[1, 2])

        # x = exp(s) * z + b
        # log p(d-u, v | c_text) = log p(x | c_text) + log|det dx/dz dr/d(d-u)|
        # dx/dz = exp(s) -> log|det dx/dz| = sum log exp(s) = sum s
        x = z * torch.exp(self.prior_log_scale) + self.prior_shift
        log_det = torch.sum(self.prior_log_scale.expand(b, -1, t), dim=[1, 2])
        log_det_p = log_det_p + log_det

        # x_k = f(x_k-1) (k=1...n)
        # log p(d-u, v | c_text) = log p(x_k | c_text) + log|det dx_k/dx_k-1 ... dx/dz dr/d(d-u)|
        for flow in self.prior_layers:
            x, log_det = flow(x, g=c_text, reverse=False)
            x = torch.flip(x, [1])
            log_det_p = log_det_p + log_det

        # log p(d-u, v | c_text) = log N(x_k; 0, I) + log|det dx_k/dx_k-1 ... dx/dz dr/d(d-u)|
        log_eps = torch.sum((-0.5 * (math.log(2 * math.pi) + x.pow(2))), dim=[1, 2])
        log_p = log_eps + log_det_p
        return log_p

    def forward(
        self,
        x: torch.Tensor,
        d: torch.Tensor | None = None,
        g: torch.Tensor | None = None,
        noise_scale: float = 1.0,
        reverse: bool = False,
    ) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Input tensor (batch_size, input_size, seq_length).
            d (torch.Tensor | None): Duration tensor (batch_size, 1, seq_length).
            g (torch.Tensor | None): Optional conditioning tensor (batch_size, cond_size, seq_length).
            noise_scale (float): Sampling noise scale.
            reverse (bool): If True, sample log(d-u) else return lower bound.

        Returns:
            torch.Tensor: Sampled log(d-u) (batch_size, 1, seq_length) else lower bound (batch_size,).
        """
        b, _, t = x.shape
        c_text = self.text_condition_encoder(x, g=g)  # (batch_size, hidden_size, seq_length)

        if reverse:
            # log(d-u)
            r = self._prior_forward(c_text, noise_scale=noise_scale, reverse=True)  #  (batch_size, 1, seq_length)
            return r

        if d is None:
            raise ValueError("d must be provided when reverse=False.")

        h_d = self.duration_condition_encoder(d)
        x = c_text + h_d

        noise = torch.randn(b, 2, t, device=x.device, dtype=x.dtype)
        u, v, log_q = self._posterior_forward(x, noise)

        z = torch.cat([torch.log((d - u).clamp(min=1e-10)), v], dim=1)
        log_p = self._prior_forward(c_text, z=z, noise_scale=noise_scale, reverse=False)  # (batch_size,)

        # log p(d-u, v | c_text) - log q(u, v | d, c_text)
        lower_bound = log_p - log_q  # (batch_size,)

        return lower_bound
