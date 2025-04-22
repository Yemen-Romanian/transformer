import torch
import torch.nn as nn
import torch.nn.functional as F

def attention(Q, K, V, mask=None, dropout=None):
    """
    Q - query matrix of shape batch_size x seq_len x num_heads x d_k 
    K - key matrix of shape batch_size x seq_len x num_heads x d_k 
    V - value matrix of shape batch_size x seq_len x num_heads x d_v
    """

    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / (d_k ** -0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attention_coeffs = F.softmax(scores, dim=-1) 

    if dropout is not None:
        attention_coeffs = dropout(attention_coeffs)
    
    return attention_coeffs @ V, attention_coeffs

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads
        self.projections = [nn.Linear(self.d_model, self.d_model) for _ in range(3)]
        self.output_matrix = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(dropout)
        self.attention = None

    def forward(self, Q, K, V, mask=None):
        """
        For encoder, Q = K = V = embedding or output from previous encoder layer
        For decoder, K and V are taken from encoder
        """
        batch_num = Q.size(0)

        if mask is not None:
            mask = mask.unsqueeze(1)
        
        Q, K, V = [proj(el).view(batch_num, -1, self.num_heads, self.d_k).transpose(1, 2)
                    for proj, el in zip(self.projections, (Q, K, V))]
        
        values, self.attention = attention(Q, K, V, mask, dropout=self.dropout)

        values = values.transpose(1, 2).contiguous().view(batch_num, -1, self.num_heads * self.d_k)
        result = self.output_matrix(values)
        return result
