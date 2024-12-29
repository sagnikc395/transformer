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


# multi-head attention block
# input sequnce -> convert into 3 matrices (Q,K,V) of the same size
# multiply these by matrices (WQ,WR,WT) and gives a resultant matrix
# and combined the heads matrix of the resultant matrix


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "require d_model to be divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model)  # Wq
        self.w_k = nn.Linear(d_model, d_model)  # Wk
        self.w_v = nn.Linear(d_model, d_model)  # Wv

        self.w_o = (nn.Linear(d_model, d_model),)  # Wo

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # q -> matrix multiplication
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        # apply the mask and replace them with very very small values
        if mask is not None:
            # set the value with a small value,for each mask is 0
            attention_scores.masked_fill_(mask == 0, -1e9)

        # applying softmax
        attention_scores = attention_scores.softmax(
            dim=-1
        )  # (batch,h,seq_len,seq_len)
        # set dropout if present
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # mask is if we want some words not interacting with other words
        # we mask them
        query = self.w_q(
            q
        )  # (batch,seq_len,d_model) ---> (batch,seq_len,d_model)
        key = self.w_k(
            k
        )  # (batch,seq_len,d_model) ---> (batch,seq_len,d_model)
        value = self.w_v(
            v
        )  # (batch,seq_len,d_model) ---> (batch,seq_len,d_model)

        # divide into small matrices for each head
        # we add a transpose cause we have prefer to have the second dimension
        # this way all the head will see all the dimension

        # (batch,seq_len,d_model) -> (batch,seq_len,h,d_k) ->(transpose) -> (batch,h,seq_len,d_k)
        query = query.view(
            query.shape[0], query.shape[1], self.h, self.d_k
        ).transpose(1, 2)
        key = key.view(
            query.shape[0], query.shape[1], self.h, self.d_k
        ).transpose(1, 2)
        value = value.view(
            query.shape[0], query.shape[1], self.h, self.d_k
        ).transpose(1, 2)

        # now calculate the attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )

        # combining all the heads together
        # (batch,h,seq_len,d_k) -> (batch,seq_len,h,d_k) -> (batch,seq_len,d_model)
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(x.shape[0], -1, self.h * self.d_k)
        )

        # multiply result by Wo
        # (batch,seq_len,d_model) -> (batch,seq_len,d_model)
        return self.w_o(x)
