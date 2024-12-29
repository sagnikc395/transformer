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

    # add the positional encoding to every word in
    # the sentence
    def forward(self, x):
        # since the positional encoding is fixed,
        # we set learn_grad as false
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(
            False
        )  # (batch, seq_len, d_model)
        return self.dropout(x)


# Layer Normalization model
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        # we need eps as if sigma -> 0 , then we dont want very big numbers
        # or very small numbers and to make this numerically stable.
        self.alpha = nn.Parameter(torch.ones(1))  # multiplied
        self.bias = nn.Parameter(torch.zeros(1))  # added

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


# feed forward model
# fully connected layer that the model uses for both encoder and decoder
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_l = nn.Linear(d_model, d_ff)  # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_l = nn.Linear(d_ff, d_model)  # W2 and B2

    def forward(self, x):
        # (Batch,seq_len,d_model) --> (via linear_l) -> (Batch,seq_len,d_ff) -> (relu)
        # -> (batch,seq_len,d_model)
        return self.linear_l(self.dropout(torch.relu(self.linear_l(x))))

    