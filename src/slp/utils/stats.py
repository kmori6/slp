import numpy as np
import torch


def tensor_to_float_stats(stats: dict[str, torch.Tensor]) -> dict[str, float]:
    """Converts a dictionary of torch.Tensor stats to a dictionary of float stats."""
    return {k: v.item() for k, v in stats.items()}


def average_stats(stats_list: list[dict[str, float]]) -> dict[str, float]:
    """Averages a list of stats dictionaries."""
    if not stats_list:
        raise ValueError("stats_list cannot be empty")

    keys = stats_list[0].keys()
    avg_stats: dict[str, float] = {}
    for key in keys:
        avg_stats[key] = float(np.mean([stats[key] for stats in stats_list]))

    return avg_stats
