from nebula.model.ar_net import arNet
from nebula.common import beam_search
from nebula import root
import os.path as osp


def compare(output, label):

    output = output[1:-1]
    label = label[1:-1]

    size = max(len(output), len(label))

    match = 0
    for i in range(size):
        if i >= len(output) or i >= len(label):
            continue

        if output[i] == label[i]:
            match += 1

    return match, size


trained_model_path = "C:/Users/aphri/Documents/t0002/pycharm/data/ar_fps6_gray_scale2/model/model_best.pt"
df_path = "C:/Users/aphri/Documents/t0002/pycharm/data/ar_fps6_gray_scale2/df.pkl"
model = arNet(
    df_path=df_path,
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
    trg = batch[2] # [2, 0-9, 3] # [2, 1,10, 3]

    res = beam_search(model.model, src, src_mask, sos_id, eos_id, None)
    match, size = compare(res[0].tolist(), trg[0].tolist())
    correct_count += match
    total_count += size
    # -2 for removing sos and eos tokens

    print(f"current accuracy: {correct_count / total_count}")

accuracy = correct_count / total_count
print(f"accuracy: {accuracy}")
