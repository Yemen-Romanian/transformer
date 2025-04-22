import torch.nn as nn
import torch
import math
import matplotlib.pyplot as plt

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        assert d_model > 0

        pos_indices = torch.arange(0, max_len).unsqueeze(1)
        embedding_indices = torch.arange(0, d_model, 2)

        pe = torch.zeros((max_len, d_model))
        div_term = torch.exp(embedding_indices * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos_indices * div_term)
        pe[:, 1::2] = torch.cos(pos_indices * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x - input embedding of shape (batch_size x sequence_len x d_model)
        """
        return self.dropout(x + self.pe[:, :x.size(1)])

