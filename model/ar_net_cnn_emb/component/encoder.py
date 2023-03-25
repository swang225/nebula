import torch
import torch.nn as nn
from .sublayer import MultiHeadAttentionLayer, PositionwiseFeedforwardLayer
from .convolution import ConvolutionLayer
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self,
                 embedding_shape,
                 hid_dim,  # == d_model
                 n_layers,
                 n_heads,
                 pf_dim,
                 dropout,
                 device,
                 max_length=128,
                 embedding_size=256
                 ):
        super().__init__()

        self.device = device

        self.conv_layer = ConvolutionLayer(
            embedding_shape=embedding_shape,
            output_embedding_len=embedding_size
        )

        self.idx_embedding = nn.Embedding(embedding_size, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

        self.fc = nn.Linear(embedding_size, hid_dim)

    def forward(self, src, src_mask):

        src = self.conv_layer(src)
        src = self.fc(src)

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        # for video frames the src is already in embeddings format
        src = self.dropout(src + self.pos_embedding(pos))
        # src = [batch size, src len, hid dim]

        for layer in self.layers:
            src, enc_attention = layer(src, src_mask)

        # src = [batch size, src len, hid dim]
        return src, enc_attention


class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, src len]

        # self attention
        _src, _attention = self.self_attention(
            src, src, src, src_mask.unsqueeze(1).unsqueeze(2))

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        # position-wise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]
        # print('EncoderLayer->forward:', src)
        return src, _attention