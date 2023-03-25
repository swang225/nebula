import numpy as np
import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    '''
    A transformer-based Seq2Seq model.
    '''
    def __init__(self,
                 encoder,
                 decoder,
                 trg_pad_idx,
                 device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask
        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, src_mask, trg):
        # src = [batch size, src len]
        # src_mask = [batch size, 1, 1, src len]
        # trg = [batch size, trg len]

        trg_mask = self.make_trg_mask(trg)
        # trg_mask = [batch size, 1, trg len, trg len]

        enc_src, enc_attention = self.encoder(src, src_mask)

        # enc_src = [batch size, src len, hid dim]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]

        return output, attention