import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchvision.transforms as transforms


def split_df(df, r1=0.8, r2=0.5, seed=123):
    np.random.seed(seed)
    mask = np.random.rand(len(df)) < r1

    train_df = df[mask]
    valid_test_df = df[~mask]

    mask = np.random.rand(len(valid_test_df)) < r2
    valid_df = valid_test_df[mask]
    test_df = valid_test_df[~mask]

    return train_df, test_df, valid_df


def split_ar_anim_df(df):
    df1 = df[df["count"] == 1]
    df2 = df[df["count"] == 2]
    df3 = df[df["count"] == 3]

    train_list = []
    test_list = []
    valid_list = []
    for curr_df in [df1, df2, df3]:
        train_df, test_df, valid_df = split_df(curr_df)
        train_list.append(train_df)
        test_list.append(test_df)
        valid_list.append(valid_df)

    train_df = pd.concat(train_list, axis=0)
    test_df = pd.concat(test_list, axis=0)
    valid_df = pd.concat(valid_list, axis=0)

    return train_df, test_df, valid_df


def build_vocab(labels):
    res = []
    for label in labels:
        res += [label]

    vocab = build_vocab_from_iterator(
        res,
        specials=['<unk>', '<pad>', '<sos>', '<eos>'],
        min_freq=2
    )

    return vocab


class ArAnimDataset(Dataset):
    def __init__(
            self,
            data,
            label_vocab,
    ):

        data = data.reset_index(drop=True)
        data.columns = ["source", "label"]
        self.data = data

        self.label_vocab = label_vocab

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.6833,), (0.2241,))  # mean and standard deviation from first image
            ]
        )

    def string_to_ids(self, input, vocab):

        stoi = vocab.get_stoi()
        res = [stoi[t] for t in input]

        res = [stoi["<sos>"]] + res + [stoi["<eos>"]]
        return res

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        cur_data = self.data.loc[idx]

        src = torch.tensor([self.transform(i).tolist()[0] for i in cur_data["source"]])

        lbl = self.string_to_ids(cur_data["label"], self.label_vocab)

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


def get_embedding_shape(df):
    res = df["embedding"][0][0].shape

    return res


def setup_data(df_path, batch_size, random_seed=1):
    df = pd.read_pickle(df_path)
    es = get_embedding_shape(df)

    train_df, test_df, valid_df = split_ar_anim_df(df)
    train_df = train_df.sample(frac=1, random_state=random_seed) # randomize train_df

    label_vocab = build_vocab(df["label"])

    train_ds = ArAnimDataset(train_df[["embedding", "label"]], label_vocab=label_vocab)
    train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            collate_fn=DataPadder(
                trg_pad_id=label_vocab.get_stoi()['<pad>'],
                embedding_shape=es
            )
        )

    train_ds_small = ArAnimDataset(train_df[["embedding", "label"]].head(1000), label_vocab=label_vocab)
    train_dl_small = DataLoader(
        train_ds_small,
        batch_size=batch_size,
        collate_fn=DataPadder(
            trg_pad_id=label_vocab.get_stoi()['<pad>'],
            embedding_shape=es
        )
    )

    test_ds = ArAnimDataset(test_df[["embedding", "label"]], label_vocab=label_vocab)
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        collate_fn=DataPadder(
            trg_pad_id=label_vocab.get_stoi()['<pad>'],
            embedding_shape=es
        )
    )

    validation_ds = ArAnimDataset(valid_df[["embedding", "label"]], label_vocab=label_vocab)
    validation_dl = DataLoader(
        validation_ds,
        batch_size=batch_size,
        collate_fn=DataPadder(
            trg_pad_id=label_vocab.get_stoi()['<pad>'],
            embedding_shape=es
        )
    )

    return train_dl, validation_dl, test_dl, train_dl_small, label_vocab, es


if __name__ == '__main__':
    df_path = "C:/Users/aphri/Documents/t0002/pycharm/data/ar_fps6_gray_scale2/df_2d.pkl"
    batch_size = 10

    train_dl, validation_dl, test_dl, train_dl_small, label_vocab, embedding_len = setup_data(df_path, batch_size)

    for d in train_dl_small:
        print(d)

    print(label_vocab.get_stoi())
