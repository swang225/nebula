from nebula.model.mnist_net_cnn_emb import mnistNetCNNEMB
from nebula.common import one_step_search
from nebula import root
import os.path as osp


trained_model_path = osp.join(
    root(), "model", "mnist_net_cnn_emb", "saved_models_1s", "model_best.pt"
)
model = mnistNetCNNEMB(
    batch_size=1,
    location="C:/Users/aphri/Documents/t0002/pycharm/data/mnist/download_data"
)
model.load_model(trained_model_path=trained_model_path)

dl = model.test_dl
sos_id = model.label_vocab.get_stoi()["<sos>"]
eos_id = model.label_vocab.get_stoi()["<eos>"]

correct_count = 0
total_count = 0
for i, batch in enumerate(dl):
    src = batch[0]
    src_mask = batch[1]
    trg = batch[2] # [2, 0-9, 3] # [2, 1,10, 3]

    res = one_step_search(model.model, src, src_mask, sos_id)
    cres = trg[0][1] == res[0][0]
    correct_count += cres.count_nonzero().tolist()
    total_count += 1

    print(f"current accuracy: {correct_count / total_count}")

accuracy = correct_count / total_count
print(f"accuracy: {accuracy}")
