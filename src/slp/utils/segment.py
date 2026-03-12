import torch


def slice_segments(x: torch.Tensor, start_indices: torch.Tensor, segment_size: int) -> torch.Tensor:
    """Slice fixed-size segments from a batched time-series tensor.

    Args:
        x (torch.Tensor): Input tensor with shape (batch_size, channels, time_length).
        start_indices (torch.Tensor): Start indices for each batch item with shape (batch_size,).
        segment_size (int): Number of frames to extract.

    Returns:
        torch.Tensor: Segment tensor with shape (batch_size, channels, segment_size).
    """
    b, c, t = x.shape
    y = torch.zeros(b, c, segment_size, device=x.device, dtype=x.dtype)
    for i in range(b):
        s = int(start_indices[i].item())
        e = min(s + segment_size, t)
        if e > s:
            y[i, :, : e - s] = x[i, :, s:e]
    return y
