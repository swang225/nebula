from nebula.common import read_pickle
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader
import os.path as osp
import numpy as np
import torch
import os
import random


def to_actions(f):
    words = f.split("_")
    return [words[1]]

def build_vocab_from_words(words):
    labels = [[w] for w in words]

    vocab = build_vocab_from_iterator(
        labels,
        specials=['<unk>', '<pad>', '<sos>', '<eos>'],
        min_freq=1
    )

    return vocab


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
            max_length,
    ):
        self.pickle_dir = pickle_dir
        self.files = files
        self.label_vocab = label_vocab
        self.max_length = max_length

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
        frames = frames[:self.max_length]
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
    def pad_frame(embedding_shape, frame):
        return np.array([[0] * embedding_shape[1]] * embedding_shape[0])

    @staticmethod
    def comform(data, shape):
        x, y, z = shape
        data = data[:x, :y, :z] # remove excess

        dx, dy, dz = [t[0]-t[1] for t in zip(shape, data.shape)]
        data = np.pad(
            data,
            pad_width=((0, dx), (0, dy), (0, dz)),
            constant_values=0
        )
        return data

    @staticmethod
    def batch_src(data, embedding_shape):
        data_len_max = max([len(d) for d in data])

        shape = [data_len_max] + list(embedding_shape)
        data_batch = torch.tensor([
            DataPadder.comform(d, shape)
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


def split_file_dict(file_dict, frac=0.8, limit=None, nclasses=None, seed=111):

    nclasses = nclasses if nclasses is not None else len(file_dict)
    limit = limit if limit is None else int(limit / nclasses)

    count = 0
    train_files = []
    test_files = []
    valid_files = []
    classes = []
    for k, files in file_dict.items():
        if nclasses is not None and count >= nclasses:
            break

        if limit is not None:
            files = files[:limit]

        (
            cur_train_files,
            cur_test_files,
            cur_valid_files
        ) = split_files(files, frac=frac)

        train_files += cur_train_files
        test_files += cur_test_files
        valid_files += cur_valid_files
        classes.append(k)

        count += 1

    random.seed(seed)
    random.shuffle(train_files)
    random.shuffle(test_files)
    random.shuffle(valid_files)

    return train_files, test_files, valid_files, classes


def get_file_dict(dir, format="mp4"):
    res = {}
    def add_file(action, file):
        if action not in res:
            res[action] = []
        res[action].append(file)

    for filename in os.listdir(dir):
        if filename.split(".")[-1] == format:

            action = to_actions(filename)[0]
            add_file(action, filename)

    return res


def setup_data(
        batch_size=10,
        base_dir="C:/Users/aphri/Documents/t0002/pycharm/data/ucf101/pickle_fps6_scale5",
        limit=1000,
        nclasses=10,
        max_length=128,
        embedding_shape=(120, 160)
):
    file_dict = get_file_dict(dir=base_dir, format="pkl")
    (
        train_files,
        test_files,
        valid_files,
        classes
    ) = split_file_dict(file_dict, limit=limit, nclasses=nclasses)

    label_vocab = build_vocab_from_words(classes)

    train_ds = UCF101Dataset(
        pickle_dir=base_dir,
        files=train_files,
        label_vocab=label_vocab,
        max_length=max_length
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        collate_fn=DataPadder(
            trg_pad_id=label_vocab.get_stoi()['<pad>'],
            embedding_shape=embedding_shape
        )
    )

    train_ds_small = UCF101Dataset(
        pickle_dir=base_dir,
        files=train_files[:100],
        label_vocab=label_vocab,
        max_length=max_length
    )
    train_dl_small = DataLoader(
        train_ds_small,
        batch_size=batch_size,
        collate_fn=DataPadder(
            trg_pad_id=label_vocab.get_stoi()['<pad>'],
            embedding_shape=embedding_shape
        )
    )

    test_ds = UCF101Dataset(
        pickle_dir=base_dir,
        files=test_files,
        label_vocab=label_vocab,
        max_length=max_length
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        collate_fn=DataPadder(
            trg_pad_id=label_vocab.get_stoi()['<pad>'],
            embedding_shape=embedding_shape
        )
    )

    validation_ds = UCF101Dataset(
        pickle_dir=base_dir,
        files=valid_files,
        label_vocab=label_vocab,
        max_length=max_length
    )
    validation_dl = DataLoader(
        validation_ds,
        batch_size=batch_size,
        collate_fn=DataPadder(
            trg_pad_id=label_vocab.get_stoi()['<pad>'],
            embedding_shape=embedding_shape
        )
    )

    return train_dl, validation_dl, test_dl, train_dl_small, label_vocab
