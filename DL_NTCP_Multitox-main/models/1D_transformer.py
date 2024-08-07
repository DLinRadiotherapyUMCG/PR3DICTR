import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, num_tokens, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, embed_dim)
        #self.dim = dim
    def forward(self, x):
        return self.embedding(x)
    
class PositionalEncoding(nn.Module):
    def __init__(self, seq_length, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        pe = torch.zeros(seq_length, embed_dim)
        position = torch.arange(0, seq_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]