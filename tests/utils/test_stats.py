import pytest
import torch

from slp.utils.stats import average_stats, tensor_to_float_stats


def test_tensor_to_float_stats():
    stats = {"acc": torch.tensor(0.8), "loss": torch.tensor(0.5)}
    expected = {"acc": 0.8, "loss": 0.5}

    float_stats = tensor_to_float_stats(stats)

    assert float_stats == pytest.approx(expected)


def test_average_stats():
    stats_list = [
        {"acc": 0.8, "loss": 0.5},
        {"acc": 0.9, "loss": 0.4},
    ]
    expected = {"acc": 0.85, "loss": 0.45}

    avg_stats = average_stats(stats_list)

    assert avg_stats == pytest.approx(expected)
