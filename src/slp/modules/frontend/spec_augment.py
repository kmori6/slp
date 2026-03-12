import random

import torch
import torch.nn as nn


class SpecAugment(nn.Module):
    """SpecAugment: A simple data augmentation method for automatic speech recognition.

    Proposed in D. S. Park et al., "SpecAugment: A simple data augmentation method for automatic speech recognition,"
    in Interspeech, 2019, pp. 2613-2617.

    """

    def __init__(self, num_freq_masks: int, num_time_masks: int, max_freq_mask_size: int, max_time_mask_size: int):
        super().__init__()
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.max_freq_mask_size = max_freq_mask_size
        self.max_time_mask_size = max_time_mask_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        NOTE: Time warping is movebed because it contributes, but is not a major factor in improving performance
        (Section 5, Discussion). It is the most expensive and least influential augmentation.

        Args:
            x (torch.Tensor): Input spectrogram (batch_size, frame_length, n_mels).

        Returns:
            torch.Tensor: Augmented spectrogram (batch_size, frame_length, n_mels).
        """
        if not self.training:
            return x

        _, frame_length, n_mels = x.shape

        # Frequency masking: f consecutive mel frequency channels [f0, f0+f) are masked
        for _ in range(self.num_freq_masks):
            if f := random.randint(0, self.max_freq_mask_size):
                f0 = random.randint(0, n_mels - f)
                x[:, :, f0 : f0 + f] = 0.0

        # Time masking:  t consecutive time steps [t0, t0+t) are masked
        for _ in range(self.num_time_masks):
            if t := random.randint(0, self.max_time_mask_size):
                t0 = random.randint(0, frame_length - t)
                x[:, t0 : t0 + t, :] = 0.0

        return x
