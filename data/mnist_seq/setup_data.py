import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def build_vocab():

    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    vocab = build_vocab_from_iterator(
        labels,
        specials=['<unk>', '<pad>', '<sos>', '<eos>'],
        min_freq=1
    )

    return vocab


class MNISTDataset(Dataset):
    def __init__(
            self,
            dataset,
            label_vocab,
    ):
        self.dataset = dataset

        self.label_vocab = label_vocab

    def string_to_ids(self, input, vocab):

        stoi = vocab.get_stoi()
        res = [stoi[t] for t in input]

        res = [stoi["<sos>"]] + res + [stoi["<eos>"]]
        return res

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        src, label = self.dataset[idx]

        lbl = self.string_to_ids([str(label)], self.label_vocab)

        return src, lbl


class DataPadder:
    def __init__(self, trg_pad_id, embedding_shape):
        self._trg_pad_id = trg_pad_id
        self._embedding_shape = embedding_shape

    @staticmethod
    def pad_frame(embedding_shape):
        return [[0] * embedding_shape[1]] * embedding_shape[0]

    @staticmethod
    def batch_src(data, embedding_shape):
        data_len_max = max([len(d) for d in data])

        # make 4 duplicate copies of the image, for testing
        # data_batch = torch.tensor([
        #     (d.tolist() + [DataPadder.pad_frame(embedding_shape)] * (data_len_max - len(d))) * 4
        #     for d
        #     in data
        # ])
        data_batch = torch.tensor([
            d.tolist() + [DataPadder.pad_frame(embedding_shape)] * (data_len_max - len(d))
            for d
            in data
        ])
        data_mask = torch.tensor([
            [1] * len(d) + [0] * (data_len_max - len(d))
            for d
            in data
        ])
        return data_batch, data_mask

    @staticmethod
    def batch_lbl(data, pad_id):
        data_len_max = max([len(s) for s in data])
        data_batch = torch.tensor([
            d + [pad_id] * (data_len_max - len(d))
            for d
            in data
        ])

        return data_batch

    def __call__(self, data):

        data_zip = list(zip(*data))

        src_batch, src_mask = self.batch_src(data_zip[0], self._embedding_shape)
        src_batch = src_batch.to(torch.float32)
        lbl_batch = self.batch_lbl(data_zip[1], self._trg_pad_id)
        return src_batch, src_mask, lbl_batch


def setup_data(batch_size, location="~/data"):

    train_dataset = torchvision.datasets.MNIST(
        location,
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]
        )
    )

    test_dataset = torchvision.datasets.MNIST(
        location,
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ]
        )
    )

    label_vocab = build_vocab()

    train_ds = MNISTDataset(train_dataset, label_vocab=label_vocab)
    train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            collate_fn=DataPadder(
                trg_pad_id=label_vocab.get_stoi()['<pad>'],
                embedding_shape=(28, 28)
            )
        )

    train_ds_small = MNISTDataset(train_dataset, label_vocab=label_vocab)
    train_dl_small = DataLoader(
        train_ds_small,
        batch_size=batch_size,
        collate_fn=DataPadder(
            trg_pad_id=label_vocab.get_stoi()['<pad>'],
            embedding_shape=(28, 28)
        )
    )

    test_ds = MNISTDataset(test_dataset, label_vocab=label_vocab)
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        collate_fn=DataPadder(
            trg_pad_id=label_vocab.get_stoi()['<pad>'],
            embedding_shape=(28, 28)
        )
    )

    validation_ds = MNISTDataset(test_dataset, label_vocab=label_vocab)
    validation_dl = DataLoader(
        validation_ds,
        batch_size=batch_size,
        collate_fn=DataPadder(
            trg_pad_id=label_vocab.get_stoi()['<pad>'],
            embedding_shape=(28, 28)
        )
    )

    return train_dl, validation_dl, test_dl, train_dl_small, label_vocab, (28, 28)


if __name__ == '__main__':
    batch_size = 10

    train_dl, validation_dl, test_dl, train_dl_small, label_vocab, embedding_len = setup_data(batch_size)

    print(label_vocab)
