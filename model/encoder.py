import torch.nn as nn

from attention import MultiHeadAttention
from feed_forward import FeedForward
from layer_norm import LayerNorm

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.feed_forward = FeedForward(d_model)
        self.layer_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.layer_norm(x)
        x = x + self.dropout(self.self_attention(x, x, x, mask))
        result = x + self.dropout(self.feed_forward(self.layer_norm(x)))
        return result
    

class TransformerEncoder(nn.Module):
    def __init__(self, N, d_model, num_heads, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model=d_model, num_heads=num_heads, dropout=dropout)
              for _ in range(N)]
        )
        self.norm = LayerNorm(features=d_model)

    def forward(self, x, mask):
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return self.norm(x)
