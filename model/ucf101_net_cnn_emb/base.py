import pandas as pd
import torch
import sqlite3
import re
import os
import os.path as osp

from nebula.common import get_device
from nebula.data.ucf101.setup_data import setup_data
from nebula.model.ucf101_net_cnn_emb.component.encoder import Encoder
from nebula.model.ucf101_net_cnn_emb.component.decoder import Decoder
from nebula.model.ucf101_net_cnn_emb.component.seq2seq import Seq2Seq


class Ucf101NetCNNEMB:
    def __init__(
            self,
            trained_model_path=None,
            batch_size=128,
            embedding_shape=(120, 160),
            base_dir="C:/Users/aphri/Documents/t0002/pycharm/data/ucf101/pickle_fps6_scale5",
            limit=1000,
            nclasses=10
    ):
        self.device = get_device()
        self.embedding_shape = embedding_shape

        MAX_LENGTH = 128

        (
            self.train_dl,
            self.validation_dl,
            self.test_dl,
            self.train_dl_small,
            self.label_vocab
        ) = setup_data(
            batch_size=batch_size,
            base_dir=base_dir,
            limit=limit,
            nclasses=nclasses,
            max_length=MAX_LENGTH
        )

        OUTPUT_DIM = len(self.label_vocab.vocab)
        HID_DIM = 220  # it equals to embedding dimension
        ENC_LAYERS = 4
        DEC_LAYERS = 4
        ENC_HEADS = 10
        DEC_HEADS = 10
        ENC_PF_DIM = 440
        DEC_PF_DIM = 440
        ENC_DROPOUT = 0.1
        DEC_DROPOUT = 0.1
        EMBEDDING_SIZE = 440
        NFILTERS = 1
        NCHANNELS = 20
        KERNEL_SIZE = 5
        POOL_SIZE = 2

        enc = Encoder(
            embedding_shape=self.embedding_shape,
            hid_dim=HID_DIM,
            n_layers=ENC_LAYERS,
            n_heads=ENC_HEADS,
            pf_dim=ENC_PF_DIM,
            dropout=ENC_DROPOUT,
            device=self.device,
            max_length=MAX_LENGTH,
            embedding_size=EMBEDDING_SIZE,
            nfilters=NFILTERS,
            nchannels=NCHANNELS,
            kernel_size=KERNEL_SIZE,
            pool_size=POOL_SIZE
        )

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

    def load_model(self, trained_model_path):
        self.model.load_state_dict(
            torch.load(
                trained_model_path,
                map_location=self.device
            )
        )
        self.model.eval()
