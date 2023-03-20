import pandas as pd
import torch
import sqlite3
import re
import os
import os.path as osp

from nebula.common import get_device
from nebula.data.ar_anim.setup_data import setup_data
from nebula.model.ar_net.component.encoder import Encoder
from nebula.model.ar_net.component.decoder import Decoder
from nebula.model.ar_net.component.seq2seq import Seq2Seq


class arNet:
    def __init__(
            self,
            df_path,
            trained_model_path=None,
            batch_size=128,
    ):
        self.device = get_device()

        ENC_HEADS = 8
        (
            self.train_dl,
            self.validation_dl,
            self.test_dl,
            self.train_dl_small,
            self.label_vocab,
            self.embedding_len
        ) = setup_data(
            df_path=df_path,
            batch_size=batch_size,
            nheads=ENC_HEADS
        )

        OUTPUT_DIM = len(self.label_vocab.vocab)
        HID_DIM = self.embedding_len  # it equals to embedding dimension
        ENC_LAYERS = 3
        DEC_LAYERS = 3
        DEC_HEADS = 8
        ENC_PF_DIM = 512
        DEC_PF_DIM = 512
        ENC_DROPOUT = 0.1
        DEC_DROPOUT = 0.1
        MAX_LENGTH = 128

        enc = Encoder(
            hid_dim=HID_DIM,
            n_layers=ENC_LAYERS,
            n_heads=ENC_HEADS,
            pf_dim=ENC_PF_DIM,
            dropout=ENC_DROPOUT,
            device=self.device,
            max_length=MAX_LENGTH)

        dec = Decoder(OUTPUT_DIM,
                      HID_DIM,
                      DEC_LAYERS,
                      DEC_HEADS,
                      DEC_PF_DIM,
                      DEC_DROPOUT,
                      self.device,
                      MAX_LENGTH
                      )

        self.TRG_PAD_IDX = self.label_vocab.get_stoi()["<pad>"]

        self.model = Seq2Seq(
            enc,
            dec,
            self.TRG_PAD_IDX,
            self.device
        ).to(self.device)
