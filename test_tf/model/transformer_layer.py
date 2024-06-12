import torch.nn as nn 
import torch 
from torch.nn import functional as F 
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, in_features, proj_features, num_heads: int, down_scale_ratio = 2, resolution = 16) -> None:
        super(CustomTransformerEncoderLayer, self).__init__()
        dropout = 0.0
        dim_feedforward = in_features * 2
        self.self_attn = nn.MultiheadAttention(in_features, num_heads, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(in_features, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, proj_features)

        self.norm1 = nn.LayerNorm(in_features)
        self.norm2 = nn.LayerNorm(proj_features)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_first = True

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(CustomTransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src) :

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x))
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x):
        x = self.self_attn(x, x, x)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

if __name__ == "__main__":
    X = torch.rand(64, 256, 64)
    
    b1= CustomTransformerEncoderLayer(64, 128, 4)
    b2= CustomTransformerEncoderLayer(128, 256, 4)
    X = b1(X)
    X = b2(X)
    print(X.shape)