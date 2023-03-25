from nebula.model.ucf101_net_cnn_emb import UcfNetCNNEMB
from nebula.common import beam_search
from nebula import root
import os.path as osp


trained_model_path = osp.join(
    root(), "model", "ucf101_net_cnn_emb", "saved_models", "model_best.pt"
)
model = UcfNetCNNEMB(
    batch_size=1
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
    trg = batch[2]

    res = beam_search(model.model, src, src_mask, sos_id, eos_id, None)
    cres = trg == res[0]
    correct_count += cres.count_nonzero().tolist() - 2
    total_count += cres.shape[-1] - 2
    # -2 for removing sos and eos tokens

    print(f"current accuracy: {correct_count / total_count}")

accuracy = correct_count / total_count
print(f"accuracy: {accuracy}")
