import torch


def sequence_mask(length: torch.Tensor) -> torch.Tensor:
    """Create a sequence mask.

    Args:
        length (torch.Tensor): Sequence length (batch,).

    Returns:
        torch.Tensor: Mask tensor (batch, max_length).

    Examples:
        >>> sequence_mask(torch.tensor([3, 5]))
        tensor([[ True,  True,  True, False, False],
                [ True,  True,  True,  True,  True]])
    """
    return torch.arange(length.max(), device=length.device)[None, :] < length[:, None]


def causal_mask(length: torch.Tensor) -> torch.Tensor:
    """Create a causal mask.

    Args:
        length (torch.Tensor): Sequence length (batch,).

    Returns:
        torch.Tensor: Mask tensor (batch, max_length, max_length).

    Examples:
        >>> causal_mask(torch.tensor([2, 3]))
        tensor([[[ True, False, False],
                [ True,  True, False],
                [ True,  True, False]],

                [[ True, False, False],
                [ True,  True, False],
                [ True,  True,  True]]])
    """
    max_len = length.max()
    idx = torch.arange(max_len, device=length.device)
    key_mask = sequence_mask(length)
    causal = idx[:, None] >= idx[None, :]
    return key_mask[:, None, :] & causal[None, :, :]


def streaming_mask(length: torch.Tensor, chunk_size: int, history_size: int) -> torch.Tensor:
    """Create a streaming mask.

    Proposed in X. Chen et al., "Developing realtime streaming transformer transducer for speech recognition
    on large-scale dataset," in ICASSP, 2021, pp. 5904-5908.

    Args:
        length (torch.Tensor): Sequence length (batch_size,).
        chunk_size (int): Chunk size.
        history_size (int): History size.

    Examples:
        >>> streaming_mask(torch.tensor([3, 5]), 2, 1)
        tensor([[ True,  True, False, False, False],
                [ True,  True, False, False, False],
                [ False,  True, True, True, False],
                [ False,  True, True, True, False],
                [ False,  False, False, True, True]])

    Returns:
        torch.Tensor: Mask tensor (max_sequence_length, max_sequence_length).
    """
    max_length = length.max()
    idx = torch.arange(max_length, device=length.device)
    chunk_id = idx // chunk_size
    start = chunk_id * chunk_size - history_size
    end = (chunk_id + 1) * chunk_size
    return (start[:, None] <= idx) & (idx < end[:, None])
