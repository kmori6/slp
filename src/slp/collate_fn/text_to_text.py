import torch
from torch.nn.utils.rnn import pad_sequence


class TextToTextCollateFn:
    def __init__(self, pad_token_id: int, ignore_token_id: int = -100):
        self.pad_token_id = pad_token_id
        self.ignore_token_id = ignore_token_id

    def __call__(self, sample_list: list[dict[str, list[list[int]]]]) -> dict[str, torch.Tensor]:
        input_ids_list, attention_mask_list = [], []
        decoder_input_ids_list, decoder_attention_mask_list, labels_list = [], [], []
        for sample in sample_list:
            input_ids = torch.tensor(sample["input_ids"], dtype=torch.long)
            attention_mask = torch.tensor(sample["attention_mask"], dtype=torch.bool)

            base_token = torch.tensor(sample["decoder_input_ids"], dtype=torch.long)
            base_attention_mask = torch.tensor(sample["decoder_attention_mask"], dtype=torch.bool)

            # shift for teacher forcing
            decoder_input_ids = base_token[:-1].clone().detach()
            decoder_attention_mask = base_attention_mask[:-1].clone().detach()
            labels = base_token[1:].clone().detach()

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            decoder_input_ids_list.append(decoder_input_ids)
            decoder_attention_mask_list.append(decoder_attention_mask)
            labels_list.append(labels)

        batch = {
            "input_ids": pad_sequence(input_ids_list, True, self.pad_token_id),
            "attention_mask": pad_sequence(input_ids_list, True, 0).bool(),
            "decoder_input_ids": pad_sequence(decoder_input_ids_list, True, self.pad_token_id),
            "decoder_attention_mask": pad_sequence(decoder_attention_mask_list, True, 0).bool(),
            "labels": pad_sequence(labels_list, True, self.ignore_token_id),
        }
        return batch
