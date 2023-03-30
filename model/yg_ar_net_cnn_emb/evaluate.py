from nebula.model.yg_ar_net_cnn_emb import ygarNetCNNEMB
from nebula.common import one_step_search, plot_results, read_pickle
from nebula import root
import os.path as osp


def show_results(
):
    results_path = osp.join(
        root(), "model", "yg_ar_net_cnn_emb", "saved_models", "train_results.pkl"
    )
    res = read_pickle(results_path)
    plot_results(res, title="ygarNet")


def evaluate(
        df_path="C:/Users/aphri/Documents/t0002/pycharm/data/yg_ar/image_df.pkl"
):
    trained_model_path = osp.join(
        root(), "model", "yg_ar_net_cnn_emb", "saved_models", "model_best.pt"
    )
    model = ygarNetCNNEMB(
        batch_size=1,
        df_path=df_path
    )
    model.load_model(trained_model_path=trained_model_path)

    dl = model.test_dl
    sos_id = model.label_vocab.get_stoi()["<sos>"]
    eos_id = model.label_vocab.get_stoi()["<eos>"]

    size = len(dl)
    correct_count = 0
    total_count = 0
    progress = 0
    for i, batch in enumerate(dl):
        src = batch[0]
        src_mask = batch[1]
        trg = batch[2] # [2, 0-9, 3] # [2, 1,10, 3]

        res = one_step_search(model.model, src, src_mask, sos_id)
        cres = trg[0][1] == res[0][0]
        correct_count += cres.count_nonzero().tolist()
        total_count += 1

        if i/size - progress >= 0.1:
            progress = i/size
            print(f"progress {int(progress*100)}%, accuracy: {100*correct_count / total_count}%")

    accuracy = correct_count / total_count
    print(f"accuracy: {accuracy}")


if __name__ == '__main__':
    evaluate()