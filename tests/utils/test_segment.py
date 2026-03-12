import torch

from slp.utils.segment import slice_segments


def test_slice_segments_extracts_expected_frames() -> None:
    x = torch.tensor(
        [
            [[0.0, 1.0, 2.0, 3.0, 4.0]],
            [[10.0, 11.0, 12.0, 13.0, 14.0]],
        ]
    )
    start_indices = torch.tensor([1, 2], dtype=torch.long)
    expected = torch.tensor(
        [
            [[1.0, 2.0]],
            [[12.0, 13.0]],
        ]
    )

    actual = slice_segments(x, start_indices, segment_size=2)

    assert torch.equal(actual, expected)


def test_slice_segments_zero_pads_when_slice_overflows_tail() -> None:
    x = torch.tensor(
        [
            [[1, 2, 3, 4, 5], [11, 12, 13, 14, 15]],
            [[6, 7, 8, 9, 10], [16, 17, 18, 19, 20]],
        ],
        dtype=torch.int64,
    )
    start_indices = torch.tensor([4, 3], dtype=torch.long)
    expected = torch.tensor(
        [
            [[5, 0, 0], [15, 0, 0]],
            [[9, 10, 0], [19, 20, 0]],
        ],
        dtype=torch.int64,
    )

    actual = slice_segments(x, start_indices, segment_size=3)

    assert torch.equal(actual, expected)
    assert actual.dtype == x.dtype
