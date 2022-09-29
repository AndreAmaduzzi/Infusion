import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

'''params'''
''''
d_model = 1024
nhead = 8 #la metà se troppo grande
nlayers = 6 #la metà se troppo grande
'''

class MappingNet(nn.Module):
    def __init__(self, d_model: int, nhead: int,
                 nlayers: int, out_flatshape: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))   # di quale dimensione deve essere il mio class token? io lo definisco cosi poi lo espando nel forwards
        self.proj = nn.Linear(d_model, out_flatshape)               # from 1024 to 2048x3 (we do not consider batch size, which is the last dim of the Tensor)
        self.init_weights()
    
    def init_weights(self) -> None:
        initrange = 0.1
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6) # from rwightman code
        self.proj.bias.data.zero_()
        self.proj.weight.data.uniform_(-initrange, initrange)
    

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, embed_size, batch_size]
        Returns:
            output Tensor of shape [seq_len, batch_size, embed_size]
        """
        src = torch.cat((self.cls_token.expand(-1, x.shape[1], -1), x), dim=0)  # x.shape[2] is the batch size
        src = self.pos_encoder(src)                                             # src [17, batch_size, 1024]
        src = self.transformer_encoder(src)
        output = self.proj(src[0,:])
        return output                                                           # output [batch_size, 6144]

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)