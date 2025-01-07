import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BillingualDataset(Dataset):
    def __init__(
        self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len
    ):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenzier_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        # convert the start of sentence token to a number
        self.sos_token = torch.tensor(
            [tokenizer_src.token_to_id(["[SOS]"])], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [tokenizer_src.token_to_id(["[EOS]"])], dtype=torch.int64
        )

        self.pad_token = torch.tensor(
            [tokenizer_src.token_to_id(["[PAD]"])], dtype=torch.int64
        )

    def __len__(self):
        # length of the dataset
        return len(self.ds)

    def __getitem__(self, index):
        # extract the original pair from huggingface ds
        src_target_pair = self.ds[index]
        src_text = src_target_pair["translation"][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]
        # convert each text into tokens and then into input ids
        # tokenizer will split each word and then it will map each word into the vocabulary
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # calc the padding tokens we need to add
        # pad the start and end of the seq token to it
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # also have to check if the seq we have choosen is enough, else throw an exception
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # one sentence will be sent to the input of the encoder
        # another one will be sent to the input of the decoder
        # and another one will be the output of the decoder target/label
        # add sos and eos to the src_text
        encoder_input = torch.concat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                # padding to add to the seq len
                torch.tensor(
                    [self.pad_token] * enc_num_padding_tokens,
                    dtype=torch.int64,
                ),
            ]
        )

        # build the decoder_input , just have sos no eos
        decoder_input = torch.concat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
            ]
        )

        # add EOs to the label (what we expect as output from the decoder)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.pos_token,
                torch.tensor(
                    self.pos_token * dec_num_padding_tokens, dtype=torch.int64
                ),
            ]
        )

        # check if the size of encoder_input is equal to that of the seq_len
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            # seq_len
            "decoder_input": decoder_input,
            # all the tokens that are not padding are ok
            "encoder_mask": (encoder_input != self.pad_token(0))
            .unsqueeze(0)
            .int(),  # (1,1,seq_len)
            "decoder_mask": (decoder_input != self.pad_token)
            .unsqueeze(0)
            .int()
            and causal_mask(
                decoder_input.size(0)
            ),  # (1,seq_len) and (1,seq_len,seq_len)
            # each word can only look at previous word and non-padding word (i.e dont want padding word to participiate in this )
            # causal mask
            "label": label,  # seq_len
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def causal_mask(size):
    # return all the values above the diagonal
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(int)
    return mask == 0
