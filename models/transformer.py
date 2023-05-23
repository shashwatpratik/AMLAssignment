import math

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class MLP(nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim,act_layer=nn.ReLU):
        super().__init__()
        self.L1 = nn.Linear( 2*in_dim,hidden_dim)
        self.act1 = act_layer()
        self.L2 = nn.Linear(hidden_dim, out_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.L1(x)
        x = self.act1(x)
        x = self.L2(x)
        x = self.softmax(x)
        return x


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

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class transformer(nn.Module):
    def __init__(self, input_dim,num_classes,hidden_dim=768,norm_layer=nn.LayerNorm,n_head=8,n_layer=8,seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.cls_token = nn.Parameter(torch.zeros(1,input_dim))
        self.ax_sep = nn.Parameter(torch.rand(1, input_dim),requires_grad=False)
        self.cor_sep = nn.Parameter(torch.rand(1, input_dim), requires_grad=False)
        self.sag_sep = nn.Parameter(torch.rand(1, input_dim), requires_grad=False)
        self.pos_encoder = PositionalEncoding(input_dim)
        encoder_layers = TransformerEncoderLayer(input_dim, n_head, hidden_dim)
        norm = norm_layer(input_dim)
        #norm = None
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layer,norm=norm)
        # 2* Input dimention because we concat cls on data which makes data dim grow by 2
        self.head = MLP(input_dim,hidden_dim,out_dim=num_classes,act_layer=nn.ReLU)

    def prepare_token(self,x):
        P, B, N, embed_dim = x.shape
        y = self.cls_token.repeat(B,1,1)
        y = torch.cat((y,x[0,:,:,:]), dim=1)
        y = torch.cat((y,self.ax_sep.repeat(B,1,1)), dim=1)
        y = torch.cat((y,x[1,:,:,:]), dim=1)
        y = torch.cat((y,self.cor_sep.repeat(B,1,1)), dim=1)
        y = torch.cat((y,x[2,:,:,:]), dim=1)
        y = torch.cat((y,self.sag_sep.repeat(B,1,1)), dim=1)
        # B, 1+(P*N)+3, embed_dim = Y.shape
        return y

    def forward(self,x):

        x = self.prepare_token(x)
        x = x.permute(1,0,2)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # 1+(P*N)+3,B,embed_dim = x.shape
        # Computing Mean after making Batch first accros all Slice except 1st(class Token)
        x = x.permute(1,0,2)
        x = torch.cat((x[:,0,:],torch.mean(x[:, 1:, :], dim=1)), dim=1)
        x = self.head(x)
        return x
