import pytest

from slp.collate_fn.text_to_text import TextToTextCollateFn


@pytest.fixture
def collate_fn() -> TextToTextCollateFn:
    return TextToTextCollateFn(pad_token_id=0, ignore_token_id=-100)


def test_collate_fn(collate_fn: TextToTextCollateFn):
    sample_list = [
        {
            "input_ids": [1, 2, 3],
            "attention_mask": [True, True, True],
            "decoder_input_ids": [4, 5, 6],
            "decoder_attention_mask": [True, True, True],
        },
        {
            "input_ids": [7, 8],
            "attention_mask": [True, True],
            "decoder_input_ids": [9, 10],
            "decoder_attention_mask": [True, True],
        },
    ]

    batch = collate_fn(sample_list)

    assert set(batch.keys()) == {
        "input_ids",
        "attention_mask",
        "decoder_input_ids",
        "decoder_attention_mask",
        "labels",
    }
    assert batch["input_ids"].shape == (2, 3)  # (batch_size, source_sequence_length)
    assert batch["attention_mask"].shape == (2, 3)  # (batch_size, source_sequence_length)
    assert batch["decoder_input_ids"].shape == (2, 2)  # (batch_size, target_sequence_length - 1)
    assert batch["decoder_attention_mask"].shape == (2, 2)  # (batch_size, target_sequence_length - 1)
    assert batch["labels"].shape == (2, 2)  # (batch_size, target_sequence_length - 1)

    # Teacher-forcing shift check
    assert batch["decoder_input_ids"][0].tolist() == [4, 5]
    assert batch["labels"][0].tolist() == [5, 6]

    # Padding check
    assert batch["input_ids"][1, 2].item() == 0
    assert batch["labels"][1, 1].item() == -100
