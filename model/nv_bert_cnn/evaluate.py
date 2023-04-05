from nebula.model.nv_bert_cnn import nvBertCNN
import torch
import torch.nn as nn
import os.path as osp
import pickle

from nebula.common import Counter, read_pickle, write_pickle

import numpy as np
import random
import time
import math


def evaluate(model, iterator):
    model.eval()

    correct = 0
    total = 0

    counter = Counter(total=len(iterator))
    counter.start()

    print("- start evaluating")
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch[0]
            trg = batch[1]

            trg_len = trg.shape[-1]

            output, _ = model(src, trg[:, :-1])

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            res = output.argmax(dim=1) == trg
            correct += res.sum().item()
            total += len(res)

            counter.update()

            print(f"currency accuracy: {correct/total}")

            del src
            del trg
            del output
            del output_dim

    print(f"final accuracy: {correct/total}")


if __name__ == '__main__':
    from nebula import root

    temp_dataset_path = "C:/Users/aphri/Documents/t0002/pycharm/data/ncnet/temp_data"
    batch_size = 1

    model = nvBertCNN(
        temp_dataset_path=temp_dataset_path,
        batch_size=batch_size
    )
    # # load the current best model
    from nebula import root
    model.load_model(osp.join(root(), "model", "nv_bert_cnn", "result", "model_best_cnn.pt"))

    evaluate(model.model, model.test_dl)
