import torch
import torch.nn as nn
from transformers import BertModel
from .sublayer import MultiHeadAttentionLayer, PositionwiseFeedforwardLayer


EMBEDDING_SIZE = 768


class BertEncoder(nn.Module):
    def __init__(
            self,
            hid_dim,  # == d_model
            n_layers,
            n_heads,
            pf_dim,
            dropout,
            device,
            TOK_TYPES,
            max_length=128
    ):
        super(BertEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.fc = nn.Linear(EMBEDDING_SIZE, hid_dim)

        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

        self.device = device

    def forward(self, input_ids, attention_mask, tok_types, batch_matrix):

        batch_size = input_ids.shape[0]
        src_len = input_ids.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        embeddings, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        embeddings = self.fc(embeddings)

        src = self.dropout((embeddings * self.scale) + self.pos_embedding(pos))
        # src = [batch size, src len, hid dim]

        for layer in self.layers:
            src, enc_attention = layer(src, attention_mask, batch_matrix)

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

    def forward(self, src, src_mask, batch_matrix):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, src len]

        # self attention
        _src, _attention = self.self_attention(src, src, src, src_mask, batch_matrix)

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
