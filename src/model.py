import torch
import torch.nn as nn
import math


# first layer -> creating the input embeddings.
class InputEmbeddings(nn.Module):
    # mapping of numbers and a vector of size 512()
    # d_model and vocab size
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # using the embedding layer by pytorch to set the embedding
        return self.embedding(x) * math.sqrt(self.d_model)


# positional encoding model
class PositionalEncoding(nn.Module):
    # seq_length -> max length for the sentence
    # dropout -> to make the model less overfit
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # build  positional encoding
        # create a matrix of shape(seq_len,d_model)
        pe = torch.zeros(seq_len, d_model)
        # from the previous formula to create positional encodings
        # using a simplified calculation using log space for numerical stability
        # create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        # apply the sin for even position and odd for odd postion
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # add the batch dimension to tensor so that we do positional encoding
        pe = pe.unsqueeze(0)  # (1,seq_lem,d_model)
        # register the positional encoding as a buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(
            False
        )  # (batch, seq_len, d_model)
        return self.dropout(x)
