import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from slp.losses.hifi_gan.discriminator_adversarial_loss import DiscriminatorAdversarialLoss
from slp.losses.hifi_gan.feature_matching_loss import FeatureMatchingLoss
from slp.losses.hifi_gan.generator_adversarial_loss import GeneratorAdversarialLoss
from slp.losses.hifi_gan.mel_spectrogram_loss import MelSpectrogramLoss
from slp.modules.hifi_gan.discriminator import Discriminator
from slp.modules.hifi_gan.generator import Generator
from slp.modules.search.monotonic_alignment_search import monotonic_alignment_search
from slp.modules.vits.flow import Flow
from slp.modules.vits.posterior_encoder import PosteriorEncoder
from slp.modules.vits.stochastic_duration_predictor import StochasticDurationPredictor
from slp.modules.vits.text_encoder import TextEncoder
from slp.utils.mask import sequence_mask
from slp.utils.segment import slice_segments


class VitsGenerator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        spec_size: int,
        segment_size: int = 32,
        hidden_size: int = 192,
        text_ffn_size: int = 768,
        text_num_heads: int = 2,
        text_num_layers: int = 6,
        text_dropout_rate: float = 0.1,
        text_window_size: int = 4,
        posterior_kernel_size: int = 5,
        posterior_dilation_rate: int = 1,
        posterior_num_layers: int = 16,
        posterior_dropout_rate: float = 0.0,
        flow_kernel_size: int = 5,
        flow_dilation_rate: int = 1,
        flow_num_blocks: int = 4,
        flow_num_layers: int = 4,
        flow_dropout_rate: float = 0.0,
        duration_kernel_size: int = 3,
        duration_dropout_rate: float = 0.5,
        duration_num_flows_blocks: int = 3,
        duration_num_prior_layers: int = 4,
        duration_num_condition_layers: int = 3,
        duration_num_posterior_layers: int = 4,
        duration_spline_num_bins: int = 10,
        duration_spline_bound: float = 5.0,
        speaker_size: int = 0,
        cond_size: int = 0,
    ):
        super().__init__()
        if speaker_size > 0 and cond_size <= 0:
            raise ValueError("cond_size must be > 0 when speaker_size > 0.")

        self.segment_size = segment_size
        self.hidden_size = hidden_size

        self.posterior_encoder = PosteriorEncoder(
            input_size=spec_size,
            output_size=hidden_size,
            hidden_size=hidden_size,
            kernel_size=posterior_kernel_size,
            dilation_rate=posterior_dilation_rate,
            num_layers=posterior_num_layers,
            dropout_rate=posterior_dropout_rate,
            cond_size=cond_size,
        )
        self.flow = Flow(
            input_size=hidden_size,
            hidden_size=hidden_size,
            kernel_size=flow_kernel_size,
            dilation_rate=flow_dilation_rate,
            num_blocks=flow_num_blocks,
            num_layers=flow_num_layers,
            dropout_rate=flow_dropout_rate,
            cond_size=cond_size,
        )
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            ffn_size=text_ffn_size,
            num_heads=text_num_heads,
            num_layers=text_num_layers,
            dropout_rate=text_dropout_rate,
            window_size=text_window_size,
        )
        self.prior_proj = nn.Conv1d(hidden_size, 2 * hidden_size, 1)

        self.duration_predictor = StochasticDurationPredictor(
            input_size=hidden_size,
            hidden_size=hidden_size,
            kernel_size=duration_kernel_size,
            dropout_rate=duration_dropout_rate,
            cond_size=cond_size,
            num_flow_blocks=duration_num_flows_blocks,
            num_prior_layers=duration_num_prior_layers,
            num_condition_layers=duration_num_condition_layers,
            num_posterior_layers=duration_num_posterior_layers,
            spline_num_bins=duration_spline_num_bins,
            spline_bound=duration_spline_bound,
        )
        self.decoder = Generator(hidden_size)

        if speaker_size > 0:
            self.speaker_embedding: nn.Embedding | None = nn.Embedding(speaker_size, cond_size)
            self.speaker_proj: nn.Conv1d | None = nn.Conv1d(cond_size, hidden_size, 1)
        else:
            self.speaker_embedding = None
            self.speaker_proj = None

        self.decoder_upsample_size = self.decoder.get_upsample_size()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        xlin: torch.Tensor,
        xlin_mask: torch.Tensor,
        start_indices: torch.Tensor,
        speech: torch.Tensor,
        speaker_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """

        Args:
            input_ids (torch.Tensor): Token index tensor (batch_size, seq_length).
            attention_mask (torch.Tensor): Token mask tensor (batch_size, seq_length).
            xlin (torch.Tensor): Acoustic feature tensor (batch_size, spec_size, frame_length).
            xlin_mask (torch.Tensor): Acoustic feature mask tensor (batch_size, frame_length).
            start_indices (torch.Tensor): Segment start indices in frame units (batch_size,).
            speech (torch.Tensor): Target waveform tensor (batch_size, sample_length).
            speaker_ids (torch.Tensor | None, optional): Speaker ID tensor (batch_size,).

        Returns:
            dict[str, torch.Tensor]: Generator output dictionary containing
                y (torch.Tensor): Target waveform segment (batch_size, 1, segment_size).
                y_hat (torch.Tensor): Generated waveform segment (batch_size, 1, segment_size).
                log_d (torch.Tensor): Duration predictor output (batch_size,).
                f_z (torch.Tensor): Flow output (batch_size, hidden_size, frame_length).
                log_std_q (torch.Tensor): Posterior log standard deviation (batch_size, hidden_size, frame_length).
                m_p (torch.Tensor): Prior mean (batch_size, hidden_size, frame_length).
                log_std_p (torch.Tensor): Prior log standard deviation (batch_size, hidden_size, frame_length).
        """

        text_lengths = attention_mask.sum(dim=1)
        frame_lengths = xlin_mask.sum(dim=1)

        # global condition embedding
        if self.speaker_embedding is not None and speaker_ids is not None:
            g = self.speaker_embedding(speaker_ids)  # (batch_size, cond_size)
            g_audio = g[:, :, None].expand(-1, -1, xlin.shape[-1])  # (batch_size, cond_size, frame_length)
            g_text = g[:, :, None].expand(-1, -1, attention_mask.shape[-1])  # (batch_size, cond_size, seq_length)
        else:
            g_audio = g_text = None

        # posterior encoder
        z, _, log_std_q = self.posterior_encoder(xlin, g=g_audio)  # (batch_size, hidden_size, frame_length)

        # decoder
        z_slice = slice_segments(z, start_indices, self.segment_size)  # (batch_size, hidden_size, segment_size)
        g_audio_slice = slice_segments(g_audio, start_indices, self.segment_size) if g_audio is not None else None
        if g_audio_slice is not None and self.speaker_proj is not None:
            z_slice = z_slice + self.speaker_proj(g_audio_slice)
        y_hat = self.decoder(z_slice)

        sample_start_indices = start_indices * self.decoder_upsample_size
        y = slice_segments(speech[:, None, :], sample_start_indices, y_hat.shape[-1]).to(y_hat.device)

        # flow
        f_z = self.flow(z, g=g_audio)  # (batch_size, hidden_size, frame_length)

        # text encoder
        h_text = self.text_encoder(input_ids, attention_mask).transpose(1, 2)  # (batch_size, hidden_size, seq_length)

        # projection
        prior_stats = self.prior_proj(h_text)  # (batch_size, 2 * hidden_size, seq_length)
        m_p, log_std_p = torch.split(prior_stats, self.hidden_size, dim=1)  # (batch_size, hidden_size, seq_length)

        # monotonic alignment search
        with torch.no_grad():
            # A = argmax_A' log p(z | c_text, A') = log N(f_z; m, std)
            precision = torch.exp(-2.0 * log_std_p)
            const_term = torch.sum(-0.5 * math.log(2.0 * math.pi) - log_std_p - 0.5 * m_p.pow(2) * precision, dim=1)
            linear_term = f_z.transpose(1, 2) @ (m_p * precision)
            quadratic_term = -0.5 * f_z.pow(2).transpose(1, 2) @ precision
            log_p = const_term[:, None, :] + linear_term + quadratic_term  # (batch_size, frame_length, seq_length)
            log_p_np = log_p.detach().cpu().numpy()

            alignment = torch.zeros_like(log_p)  # (b, t_y, t_x)
            for i in range(log_p.shape[0]):
                t_xi = int(text_lengths[i].item())
                t_yi = int(frame_lengths[i].item())
                path = monotonic_alignment_search(log_p_np[i, :t_yi, :t_xi].T).T
                alignment[i, :t_yi, :t_xi] = torch.from_numpy(path).to(device=log_p.device, dtype=log_p.dtype)

            align_mask = xlin_mask[:, :, None] & attention_mask[:, None, :]  # (batch_size, frame_length, seq_length)
            alignment = alignment.masked_fill(~align_mask, 0.0)
            d = alignment.sum(dim=1, keepdim=True)  # (batch_size, 1, seq_length)

        # gradient detach
        h_text = torch.detach(h_text)
        g_text = torch.detach(g_text) if g_text is not None else None

        # stochastic duration predictor
        log_d = self.duration_predictor(h_text, d=d, g=g_text)
        m_p = (alignment @ m_p.transpose(1, 2)).transpose(1, 2)  # (batch_size, hidden_size, frame_length)
        log_std_p = (alignment @ log_std_p.transpose(1, 2)).transpose(1, 2)  # (batch_size, hidden_size, frame_length)

        return {
            "y": y,
            "y_hat": y_hat,
            "log_d": log_d,
            "f_z": f_z,
            "log_std_q": log_std_q,
            "m_p": m_p,
            "log_std_p": log_std_p,
        }

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        speaker_ids: torch.Tensor | None = None,
        noise_scale: float = 1.0,
        length_scale: float = 1.0,
    ) -> torch.Tensor:
        """

        Args:
            input_ids (torch.Tensor): Token index tensor (batch_size, seq_length).
            attention_mask (torch.Tensor): Token mask tensor (batch_size, seq_length).
            speaker_ids (torch.Tensor | None, optional): Speaker ID tensor (batch_size,).
            noise_scale (float, optional): Sampling noise scale for latent and duration generation.
            length_scale (float, optional): Global duration scaling factor.

        Returns:
            torch.Tensor: Generated waveform tensor (batch_size, 1, sample_length).
        """

        # global condition embedding 1/2
        if self.speaker_embedding is not None and speaker_ids is not None:
            g = self.speaker_embedding(speaker_ids)  # (batch_size, cond_size)
            g_text = g[:, :, None].expand(-1, -1, attention_mask.shape[-1])  # (batch_size, cond_size, seq_length)
        else:
            g_text = None

        # text encoder
        h_text = self.text_encoder(input_ids, attention_mask).transpose(1, 2)  # (batch_size, hidden_size, seq_length)

        # projection
        prior_stats = self.prior_proj(h_text)  # (batch_size, 2 * hidden_size, seq_length)
        m_p, log_std_p = torch.split(prior_stats, self.hidden_size, dim=1)  # (batch_size, hidden_size, seq_length)

        # stochastic duration predictor
        log_d = self.duration_predictor(
            h_text, attention_mask, g=g_text, reverse=True, noise_scale=noise_scale
        )  # (batch_size, 1, seq_length)
        d = torch.exp(log_d) * length_scale
        d_ceil = torch.ceil(d).masked_fill(~attention_mask[:, None, :], 0.0)
        frame_lengths = torch.sum(d_ceil, dim=[1, 2]).long()  # (batch_size,)
        frame_mask = sequence_mask(frame_lengths)  # (batch_size, frame_length)
        align_mask = frame_mask[:, :, None] & attention_mask[:, None, :]  # (b, t_y, t_x)
        d_cumsum = torch.cumsum(d_ceil.long(), dim=-1)  # (b, 1, t_x)
        idx = torch.arange(frame_mask.shape[1], device=d_ceil.device)
        align = (idx[None, None, :] < d_cumsum.transpose(1, 2)).to(dtype=d_cumsum.dtype)  # (b, t_x, t_y)
        shift = F.pad(align, (0, 0, 1, 0))[:, :-1, :]
        align = (align - shift).transpose(1, 2)  # (b, t_y, t_x)
        align = align.masked_fill(~align_mask, 0.0)

        m_p = (align @ m_p.transpose(1, 2)).transpose(1, 2)  # (batch_size, hidden_size, frame_length)
        log_std_p = (align @ log_std_p.transpose(1, 2)).transpose(1, 2)  # (batch_size, hidden_size, frame_length)
        z_p = m_p + torch.randn_like(m_p) * torch.exp(log_std_p) * noise_scale

        # global condition embedding 2/2
        g_audio = g[:, :, None].expand(-1, -1, z_p.shape[-1]) if g is not None else None

        # flow
        z = self.flow(z_p, g=g_audio, reverse=True)  # (batch_size, hidden_size, frame_length)

        # decoder
        if g_audio is not None and self.speaker_proj is not None:
            z = z + self.speaker_proj(g_audio)
        y_hat = self.decoder(z)  # (batch_size, 1, sample_length)

        return y_hat


class Vits(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        spec_size: int,
        segment_size: int = 32,
        hidden_size: int = 192,
        text_ffn_size: int = 768,
        text_num_heads: int = 2,
        text_num_layers: int = 6,
        text_dropout_rate: float = 0.1,
        text_window_size: int = 4,
        posterior_kernel_size: int = 5,
        posterior_dilation_rate: int = 1,
        posterior_num_layers: int = 16,
        posterior_dropout_rate: float = 0.0,
        flow_kernel_size: int = 5,
        flow_dilation_rate: int = 1,
        flow_num_blocks: int = 4,
        flow_num_layers: int = 4,
        flow_dropout_rate: float = 0.0,
        duration_kernel_size: int = 3,
        duration_dropout_rate: float = 0.5,
        duration_num_flows_blocks: int = 3,
        duration_num_prior_layers: int = 4,
        duration_num_condition_layers: int = 3,
        duration_num_posterior_layers: int = 4,
        duration_spline_num_bins: int = 10,
        duration_spline_bound: float = 5.0,
        speaker_size: int = 0,
        cond_size: int = 0,
        fft_size: int = 1024,
        hop_size: int = 256,
        window_size: int = 1024,
        mel_size: int = 80,
        sample_rate: int = 22050,
        min_freq: float = 0.0,
        max_freq: float = 11025.0,
        periods: list[int] = [2, 3, 5, 7, 11],
        scales: list[int] = [1],
        negative_slope: float = 0.1,
        mel_loss_weight: float = 45.0,
        kl_loss_weight: float = 1.0,
        dur_loss_weight: float = 1.0,
        gen_adv_loss_weight: float = 1.0,
        fm_loss_weight: float = 2.0,
        disc_adv_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.generator = VitsGenerator(
            vocab_size=vocab_size,
            spec_size=spec_size,
            segment_size=segment_size,
            hidden_size=hidden_size,
            text_ffn_size=text_ffn_size,
            text_num_heads=text_num_heads,
            text_num_layers=text_num_layers,
            text_dropout_rate=text_dropout_rate,
            text_window_size=text_window_size,
            posterior_kernel_size=posterior_kernel_size,
            posterior_dilation_rate=posterior_dilation_rate,
            posterior_num_layers=posterior_num_layers,
            posterior_dropout_rate=posterior_dropout_rate,
            flow_kernel_size=flow_kernel_size,
            flow_dilation_rate=flow_dilation_rate,
            flow_num_blocks=flow_num_blocks,
            flow_num_layers=flow_num_layers,
            flow_dropout_rate=flow_dropout_rate,
            duration_kernel_size=duration_kernel_size,
            duration_dropout_rate=duration_dropout_rate,
            duration_num_flows_blocks=duration_num_flows_blocks,
            duration_num_prior_layers=duration_num_prior_layers,
            duration_num_condition_layers=duration_num_condition_layers,
            duration_num_posterior_layers=duration_num_posterior_layers,
            duration_spline_num_bins=duration_spline_num_bins,
            duration_spline_bound=duration_spline_bound,
            speaker_size=speaker_size,
            cond_size=cond_size,
        )
        self.discriminator = Discriminator(periods=periods, scales=scales, negative_slope=negative_slope)

        # loss
        self.mel_loss_weight = mel_loss_weight
        self.kl_loss_weight = kl_loss_weight
        self.dur_loss_weight = dur_loss_weight
        self.gen_adv_loss_weight = gen_adv_loss_weight
        self.fm_loss_weight = fm_loss_weight
        self.disc_adv_loss_weight = disc_adv_loss_weight
        self.mel_loss_fn = MelSpectrogramLoss(
            fft_size=fft_size,
            hop_size=hop_size,
            window_size=window_size,
            mel_size=mel_size,
            sample_rate=sample_rate,
            min_freq=min_freq,
            max_freq=max_freq,
        )
        self.gen_adv_loss_fn = GeneratorAdversarialLoss()
        self.fm_loss_fn = FeatureMatchingLoss()
        self.disc_adv_loss_fn = DiscriminatorAdversarialLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        xlin: torch.Tensor,
        xlin_mask: torch.Tensor,
        start_indices: torch.Tensor,
        speech: torch.Tensor,
        speaker_ids: torch.Tensor | None = None,
        phase: Literal["generator", "discriminator"] = "generator",
    ) -> dict[str, torch.Tensor]:
        """

        Args:
            input_ids (torch.Tensor): Token index tensor (batch_size, seq_length).
            attention_mask (torch.Tensor): Token mask tensor (batch_size, seq_length).
            xlin (torch.Tensor): Acoustic feature tensor (batch_size, spec_size, frame_length).
            xlin_mask (torch.Tensor): Acoustic feature mask tensor (batch_size, frame_length).
            start_indices (torch.Tensor): Segment start indices (batch_size,).
            speech (torch.Tensor): Target waveform tensor (batch_size, sample_length).
            speaker_ids (torch.Tensor | None): Speaker ID tensor (batch_size,).
            phase (Literal["generator", "discriminator"]): Training phase.

        Returns:
            dict[str, torch.Tensor]: Loss dictionary.
        """
        if phase not in {"generator", "discriminator"}:
            raise ValueError(f"phase must be 'generator' or 'discriminator', but got {phase!r}")

        if phase == "discriminator":
            with torch.no_grad():
                outputs = self.generator(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    xlin=xlin,
                    xlin_mask=xlin_mask,
                    start_indices=start_indices,
                    speech=speech,
                    speaker_ids=speaker_ids,
                )
            y_d_rs, _ = self.discriminator(outputs["y"])
            y_d_gs, _ = self.discriminator(outputs["y_hat"].detach())
            disc_adv_loss = self.disc_adv_loss_fn(y_d_rs, y_d_gs) * self.disc_adv_loss_weight
            return {"loss": disc_adv_loss, "disc_adv_loss": disc_adv_loss}

        outputs = self.generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            xlin=xlin,
            xlin_mask=xlin_mask,
            start_indices=start_indices,
            speech=speech,
            speaker_ids=speaker_ids,
        )

        fake_wav = outputs["y_hat"].squeeze(1)
        real_wav = outputs["y"].squeeze(1)
        mel_length = torch.full((fake_wav.shape[0],), fake_wav.shape[-1], device=fake_wav.device, dtype=torch.long)
        # NOTE: stft in mel loss computation requires float32
        mel_loss = self.mel_loss_fn(fake_wav=fake_wav.float(), real_wav=real_wav.float(), length=mel_length)
        dur_loss = outputs["log_d"].sum() / attention_mask.sum()

        precision = torch.exp(-2.0 * outputs["log_std_p"].float())
        kl = (
            outputs["log_std_p"]
            - outputs["log_std_q"]
            - 0.5
            + 0.5 * (outputs["f_z"] - outputs["m_p"]).square() * precision
        )
        kl_loss = kl.masked_fill(~xlin_mask[:, None, :], 0.0).sum() / xlin_mask.sum()

        with torch.no_grad():
            _, fmap_rs = self.discriminator(outputs["y"])
        y_d_gs, fmap_gs = self.discriminator(outputs["y_hat"])
        gen_adv_loss = self.gen_adv_loss_fn(y_d_gs)
        fm_loss = self.fm_loss_fn(fmap_rs, fmap_gs)

        loss = (
            self.mel_loss_weight * mel_loss
            + self.kl_loss_weight * kl_loss
            + self.dur_loss_weight * dur_loss
            + self.gen_adv_loss_weight * gen_adv_loss
            + self.fm_loss_weight * fm_loss
        )
        return {
            "loss": loss,
            "mel_loss": mel_loss,
            "kl_loss": kl_loss,
            "dur_loss": dur_loss,
            "gen_adv_loss": gen_adv_loss,
            "fm_loss": fm_loss,
        }

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        speaker_ids: torch.Tensor | None = None,
        noise_scale: float = 1.0,
        length_scale: float = 1.0,
    ) -> torch.Tensor:
        """

        Args:
            input_ids (torch.Tensor): Token index tensor (batch_size, seq_length).
            attention_mask (torch.Tensor): Token mask tensor (batch_size, seq_length).
            speaker_ids (torch.Tensor | None, optional): Speaker ID tensor (batch_size,).
            noise_scale (float, optional): Sampling noise scale for latent and duration generation.
            length_scale (float, optional): Global duration scaling factor.

        Returns:
            torch.Tensor: Generated waveform tensor (batch_size, 1, sample_length).
        """
        return self.generator.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            speaker_ids=speaker_ids,
            noise_scale=noise_scale,
            length_scale=length_scale,
        )
