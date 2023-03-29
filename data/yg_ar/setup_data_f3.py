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


def split_ar_anim_df(df, random_seed):

    labels = set(df["label"].values)

    dfs = []
    for label in labels:
        dfs.append(df[df["label"] == label])

    train_list = []
    test_list = []
    valid_list = []
    for curr_df in dfs:
        curr_df = curr_df.sample(frac=1, random_state=random_seed)
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
        res += [[label]]

    vocab = build_vocab_from_iterator(
        res,
        specials=['<unk>', '<pad>', '<sos>', '<eos>'],
        min_freq=2
    )

    return vocab


class ygarAnimDataset(Dataset):
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
                transforms.Normalize((0.6708,), (0.0779,)) # mean and standard deviation from first image
            ]
        )

    def string_to_ids(self, input, vocab):

        stoi = vocab.get_stoi()
        res = [stoi[input]]

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

    def __call__(self, data):

        data_zip = list(zip(*data))

        src_batch = torch.tensor([d.tolist() for d in data_zip[0]])
        src_mask = torch.tensor([[1]] * len(data_zip[0]))
        lbl_batch = torch.tensor(data_zip[1])

        return src_batch, src_mask, lbl_batch


def conform_embedding(df, nheads):
    org_embedding_len = len(df["embedding"][0][0])
    new_embedding_len = (org_embedding_len // nheads) * nheads

    if org_embedding_len == new_embedding_len:
        # nothing to do just return
        return df, org_embedding_len

    # resize the embedding to be multiples of nheads
    for i in range(len(df)):
        new_embedding = df.loc[i, "embedding"]
        df.loc[i, "embedding"] = list(np.array(new_embedding)[:, :new_embedding_len])

    return df, new_embedding_len



def setup_data(df_path, batch_size, random_seed=1):
    df = pd.read_pickle(df_path)
    embedding_shape = df["images"][0][0].shape

    train_df, test_df, valid_df = split_ar_anim_df(df, random_seed)
    train_df = train_df.sample(frac=1, random_state=random_seed) # randomize train_df

    label_vocab = build_vocab(df["label"])

    train_ds = ygarAnimDataset(train_df[["images", "label"]], label_vocab=label_vocab)
    train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            collate_fn=DataPadder()
        )

    train_ds_small = ygarAnimDataset(train_df[["images", "label"]].head(100), label_vocab=label_vocab)
    train_dl_small = DataLoader(
        train_ds_small,
        batch_size=batch_size,
        collate_fn=DataPadder()
    )

    test_ds = ygarAnimDataset(test_df[["images", "label"]], label_vocab=label_vocab)
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        collate_fn=DataPadder()
    )

    validation_ds = ygarAnimDataset(valid_df[["images", "label"]], label_vocab=label_vocab)
    validation_dl = DataLoader(
        validation_ds,
        batch_size=batch_size,
        collate_fn=DataPadder()
    )

    return train_dl, validation_dl, test_dl, train_dl_small, label_vocab, embedding_shape


if __name__ == '__main__':
    df_path = "C:/Users/aphri/Documents/t0002/pycharm/data/yg_ar/f3_df.pkl"
    batch_size = 10

    train_dl, validation_dl, test_dl, train_dl_small, label_vocab, embedding_shape = setup_data(df_path, batch_size)

    for d in train_dl_small:
        print(d)

    print(label_vocab.get_stoi())
    print(len(train_dl))
    print(len(validation_dl))
    print(len(test_dl))
