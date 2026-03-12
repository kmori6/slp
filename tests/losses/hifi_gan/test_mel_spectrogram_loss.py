import torch

from slp.losses.hifi_gan.mel_spectrogram_loss import MelSpectrogramLoss


def test_backward_gradients():
    loss_fn = MelSpectrogramLoss(
        fft_size=1024,
        hop_size=256,
        window_size=1024,
        mel_size=80,
        sample_rate=22050,
        min_freq=0.0,
        max_freq=8000.0,
    )
    fake_wav = torch.randn(2, 4096, requires_grad=True)
    real_wav = torch.randn(2, 4096, requires_grad=True)
    length = torch.tensor([4096, 4096])

    loss = loss_fn(fake_wav, real_wav, length)
    loss.backward()
    assert fake_wav.grad is not None and real_wav.grad is not None
