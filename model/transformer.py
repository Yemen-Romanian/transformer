import torch.nn as nn

from encoder import TransformerEncoder
from decoder import TransformerDecoder
from positional_encoding import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, num_encoder_layers, num_decoder_layers, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(
            N=num_encoder_layers, 
            d_model=d_model, 
            num_heads=num_heads, 
            dropout=dropout
        )
        
        self.decoder = TransformerDecoder(
            N=num_decoder_layers, 
            d_model=d_model, 
            num_heads=num_heads
        )

        self.pe = PositionalEncoding(d_model)

    def encode(self, source, source_mask):
        return self.encoder(self.pe(source), source_mask)
    
    def decode(self, memory, source_mask, target, target_mask):
        return self.decoder(self.pe(target), memory, source_mask, target_mask)
    

    def forward(self, source, target, source_mask, target_mask):
        result = self.encode(source, source_mask=source_mask)
        return self.decode(result, source_mask=source_mask, target_mask=target_mask, target=target)
                     
    