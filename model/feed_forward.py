import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, dff=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dff, d_model)
        )

    def forward(self, x):
        return self.ffn(x)
