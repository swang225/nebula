import cv2
import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from skimage.morphology import skeletonize
from skimage.util import invert
import matplotlib.pyplot as plt


def display(frame):
    # display results
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

    ax.imshow(frame, cmap=plt.cm.gray)
    ax.axis('off')

    fig.tight_layout()
    plt.show()


def display2(f1, f2, t1="image 1", t2="image 2"):
    # display results
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                             sharex=True, sharey=True)

    ax = axes.ravel()

    ax[0].imshow(f1, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title(t1, fontsize=20)

    ax[1].imshow(f2, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title(t2, fontsize=20)

    fig.tight_layout()
    plt.show()


def get_frames(path, fps=None):
    cap = cv2.VideoCapture(path)

    time_increment = (
        1 / fps
        if fps is not None
        else None
    )

    res = []
    success = 1
    sec = 0
    while success:
        if time_increment is not None:
            sec += time_increment
            cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * sec)
        success, image = cap.read()
        if success:
            res.append(image)

    return res


def save_frames(
        frames,
        dir,
        prefix="frame"
):
    for i, frame in enumerate(frames):
        cv2.imwrite(str(osp.join(dir, f"{prefix}_{i}.jpg")), frame)


def to_gray(frames):
    res = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    return res


def to_scale(frames, scale=0.5):
    res = []
    for f in frames:
        width = int(f.shape[1] * scale)
        height = int(f.shape[0] * scale)
        dim = (width, height)
        resized = cv2.resize(f, dim, interpolation=cv2.INTER_AREA)
        res.append(resized)

    return res


def read_pickle(dir):
    with open(dir, 'rb') as handle:
        b = pickle.load(handle)
    return b


def write_pickle(dir, data):
    with open(dir, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def to_pickle_name(file):
    return file.split(".")[0] + ".pkl"


def flatten(frames):
    return [f.flatten() for f in frames]


def one_to_skeleton(frame):
    (thresh, BnW_image) = cv2.threshold(frame, 125, 255, cv2.THRESH_BINARY)
    image = invert(BnW_image)
    skeleton = skeletonize(image, method='lee')

    return skeleton


def to_skeleton(frames):
    return [one_to_skeleton(f) for f in frames]


def to_fps_gray_scale(
        base_dir,
        save_dir,
        files,
        fps,
        scale,
        overwrite=False,
):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    count = len(files)
    prev_progress = 0
    for i, file in enumerate(files):
        save_file = osp.join(save_dir, to_pickle_name(file))
        if osp.exists(save_file) and not overwrite:
            continue

        path = osp.join(base_dir, file)
        res = get_frames(path, fps)
        gray = to_gray(res)
        resized_gray = to_scale(gray, scale)

        write_pickle(save_file, resized_gray)

        cur_progress = int((i + 1) * 100 / count)
        if cur_progress >= prev_progress + 2:
            print(f"progress: {cur_progress}%")
            prev_progress = cur_progress


def get_files(dir, format="mp4", limit=None):
    res = []
    for filename in os.listdir(dir):
        if limit is not None and len(res) >= limit:
            break
        if filename.split(".")[-1] == format:
            res.append(filename)
    return res


def to_actions(f):
    words = f.split("_")
    res = []
    for w in words:
        if w.startswith("y") and w[1:].isnumeric():
            break
        res.append(w)

    return res


def get_df(
        pickle_dir,
        pickle_files,
        save_path,
        flatten=True,
        skeletonize=False,
        refresh=False,
        to_action_func=to_actions
):
    if osp.exists(save_path) and not refresh:
        df = pd.read_pickle(save_path)
        return df

    count = len(pickle_files)
    prev_progress = 0

    res = []
    for i, f in enumerate(pickle_files):
        frames = read_pickle(osp.join(pickle_dir, f))
        if flatten:
            frames = flatten(frames)
        if skeletonize:
            frames = to_skeleton(frames)
        actions = to_action_func(f)
        action_count = len(actions)
        res.append((frames, actions, action_count))

        cur_progress = int((i + 1) * 100 / count)
        if cur_progress >= prev_progress + 2:
            print(f"progress: {cur_progress}%")
            prev_progress = cur_progress

    df = pd.DataFrame(
        data=dict(zip(["embedding", "label", "count"], np.transpose(res)))
    )
    df.to_pickle(save_path)

    return df

