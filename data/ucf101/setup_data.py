from nebula.common import get_files, read_pickle
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader
import os.path as osp
import numpy as np
import torch


def to_actions(f):
    words = f.split("_")
    return [words[1]]


def build_vocab(files):
    labels = []
    for f in files:
        labels.append(to_actions(f))

    vocab = build_vocab_from_iterator(
        labels,
        specials=['<unk>', '<pad>', '<sos>', '<eos>'],
        min_freq=1
    )

    return vocab


class UCF101Dataset(Dataset):
    def __init__(
            self,
            pickle_dir,
            files,
            label_vocab,
    ):
        self.pickle_dir = pickle_dir
        self.files = files
        self.label_vocab = label_vocab

    @staticmethod
    def string_to_id(label, vocab):
        return vocab.get_stoi()[label]

    @staticmethod
    def string_to_ids(label, vocab):

        label_id = UCF101Dataset.string_to_id(label, vocab)
        res = [label_id]

        stoi = vocab.get_stoi()
        res = [stoi["<sos>"]] + res + [stoi["<eos>"]]
        return res

    def __len__(self):
        return len(self.files)

    def to_data(self, idx):
        file = self.files[idx]
        frames = np.array(read_pickle(osp.join(self.pickle_dir, file)))
        label = to_actions(file)[0]
        return frames, label

    def __getitem__(self, idx):

        src, label = self.to_data(idx)

        lbl = self.string_to_ids(label, self.label_vocab)

        return src, lbl


class DataPadder:
    def __init__(self, trg_pad_id, embedding_shape=None):
        self._trg_pad_id = trg_pad_id
        self._embedding_shape = embedding_shape

    @staticmethod
    def pad_frame(embedding_shape):
        return [[0] * embedding_shape[1]] * embedding_shape[0]

    @staticmethod
    def batch_src(data, embedding_shape):
        data_len_max = max([len(d) for d in data])

        data_batch = torch.tensor([
            list(d) + [DataPadder.pad_frame(embedding_shape)] * (data_len_max - len(d))
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

        if self._embedding_shape is None:
            self._embedding_shape = data[0][0][0].shape

        src_batch, src_mask = self.batch_src(data_zip[0], self._embedding_shape)
        src_batch = src_batch.to(torch.float32)
        lbl_batch = self.batch_lbl(data_zip[1], self._trg_pad_id)
        return src_batch, src_mask, lbl_batch


def split_files(files, frac=0.8):

    size = len(files)
    split = int(size * frac)

    train_files = files[:split]
    test_files = files[split:]

    size = len(test_files)
    split = int(size * 0.5)

    valid_files = test_files[:split]
    test_files = test_files[split:]

    return train_files, test_files, valid_files


def setup_data(
        batch_size=10,
        base_dir="C:/Users/aphri/Documents/t0002/pycharm/data/ucf101/pickle_fps6_scale5",
        limit=1000,
):
    files = get_files(dir=base_dir, format="pkl", limit=limit)

    label_vocab = build_vocab(files)

    train_files, test_files, valid_files = split_files(files)

    train_ds = UCF101Dataset(
        pickle_dir=base_dir,
        files=train_files,
        label_vocab=label_vocab,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        collate_fn=DataPadder(trg_pad_id=label_vocab.get_stoi()['<pad>'],)
    )

    train_ds_small = UCF101Dataset(
        pickle_dir=base_dir,
        files=train_files[:100],
        label_vocab=label_vocab,
    )
    train_dl_small = DataLoader(
        train_ds_small,
        batch_size=batch_size,
        collate_fn=DataPadder(trg_pad_id=label_vocab.get_stoi()['<pad>'],)
    )

    test_ds = UCF101Dataset(
        pickle_dir=base_dir,
        files=test_files,
        label_vocab=label_vocab,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        collate_fn=DataPadder(trg_pad_id=label_vocab.get_stoi()['<pad>'],)
    )

    validation_ds = UCF101Dataset(
        pickle_dir=base_dir,
        files=valid_files,
        label_vocab=label_vocab,
    )
    validation_dl = DataLoader(
        validation_ds,
        batch_size=batch_size,
        collate_fn=DataPadder(trg_pad_id=label_vocab.get_stoi()['<pad>'],)
    )

    return train_dl, validation_dl, test_dl, train_dl_small, label_vocab
