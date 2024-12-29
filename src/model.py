import torch 
import torch.nn as nn
import math 

# first layer -> creating the input embeddings.
class InputEmbeddings(nn.Module):
    # mapping of numbers and a vector of size 512()
    #d_model and vocab size 
    def __init__(self,d_model:int,vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        # using the embedding layer by pytorch to set the embedding
        return self.embedding(x) * math.sqrt(self.d_model)


# positional encoding model 