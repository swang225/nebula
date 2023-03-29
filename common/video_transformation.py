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


def get_loc_frame_from_data(data, loc=0.5):
    count = data.get(cv2.CAP_PROP_FRAME_COUNT)
    f_loc = int((count - 1) * loc)

    data.set(cv2.CAP_PROP_POS_FRAMES, f_loc)
    ret, frame = data.read()

    return frame


def get_loc_frame(path, loc=0.5):
    data = cv2.VideoCapture(path)
    return get_loc_frame_from_data(data, loc=loc)


def get_n_frames(path, n=8):
    data = cv2.VideoCapture(path)
    res = [get_loc_frame_from_data(data, loc=i / n) for i in range(n)]

    return res


def get_locs_frames(path, locs=[0.1, 0.5, 0.9]):
    data = cv2.VideoCapture(path)
    res = [get_loc_frame_from_data(data, loc=loc) for loc in locs]
    return res


def save_frame(path, frame):
    cv2.imwrite(path, frame)


def save_frames(
        frames,
        dir,
        prefix="frame"
):
    for i, frame in enumerate(frames):
        cv2.imwrite(str(osp.join(dir, f"{prefix}_{i}.jpg")), frame)


def to_gray_one(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def to_gray(frames):
    res = [to_gray_one(f) for f in frames]
    return res


def crop_one(frame, top, bottom, left, right):
    return frame[top:-bottom, left:-right]


def crop(frames, top, bottom, left, right):
    res = [crop_one(f, top, bottom, left, right) for f in frames]
    return res


def read_gray_frame(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def to_jpg_name(file):
    return file.split(".")[0] + ".jpg"


def to_pickle_name(file):
    return file.split(".")[0] + ".pkl"


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


def get_files(dir, format=None):
    res = []
    for filename in os.listdir(dir):
        if format is None or filename.split(".")[-1] == format:
            res.append(filename)
    return res


def filename_to_labels(filename):
    words = filename.split("_")

    orientation = words[-1]
    xangle = words[-2]
    yangle = words[-3]
    pants = words[-4]
    cloth = words[-6]
    hair = words[-8]
    action_type = words[-10]
    label = "_".join(words[:-10])

    return (
        orientation,
        xangle,
        yangle,
        pants,
        cloth,
        hair,
        action_type,
        label
    )


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


def convert_middle_frame(
        base_dir,
        save_dir,
        overwrite=False,
        crop=dict(top=70, bottom=30, left=50, right=50)
):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    files = get_files(base_dir)
    count = len(files)
    prev_progress = 0
    for i, file in enumerate(files):
        save_file = osp.join(save_dir, to_jpg_name(file))
        if osp.exists(save_file) and not overwrite:
            continue

        path = osp.join(base_dir, file)
        frame = to_gray_one(crop_one(get_loc_frame(path), **crop))

        save_frame(save_file, frame)

        cur_progress = int((i + 1) * 100 / count)
        if cur_progress >= prev_progress + 2:
            print(f"progress: {cur_progress}%")
            prev_progress = cur_progress


def df_middle_frame(
        image_dir,
        save_path,
):
    files = get_files(image_dir)
    count = len(files)
    print(count)
    prev_progress = 0
    res = []
    for i, f in enumerate(files):
        frame = read_gray_frame(osp.join(image_dir, f))
        (
            orientation,
            xangle,
            yangle,
            pants,
            cloth,
            hair,
            action_type,
            label
        ) = filename_to_labels(f)

        res.append(
            (
                frame,
                orientation,
                xangle,
                yangle,
                pants,
                cloth,
                hair,
                action_type,
                label
            )
        )

        cur_progress = int((i + 1) * 100 / count)
        if cur_progress >= prev_progress + 2:
            print(f"progress: {cur_progress}%")
            prev_progress = cur_progress

    df = pd.DataFrame(data=dict(zip(
        [
            "image",
            "orientation",
            "xangle",
            "yangle",
            "pants",
            "cloth",
            "hair",
            "action_type",
            "label"
        ],
        np.transpose(res)
    )))

    df.to_pickle(save_path)

    return df


def convert_locs_frames(
        base_dir,
        save_dir,
        overwrite=False,
        crop=dict(top=70, bottom=30, left=50, right=50),
        locs=[0.15, 0.5, 0.85]
):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    files = get_files(base_dir)
    count = len(files)
    prev_progress = 0
    for i, file in enumerate(files):
        save_file = osp.join(save_dir, to_pickle_name(file))
        if osp.exists(save_file) and not overwrite:
            continue

        path = osp.join(base_dir, file)
        frames = get_locs_frames(path, locs=locs)
        frames = [to_gray_one(crop_one(f, **crop)) for f in frames]

        write_pickle(save_file, frames)

        cur_progress = int((i + 1) * 100 / count)
        if cur_progress >= prev_progress + 2:
            print(f"progress: {cur_progress}%")
            prev_progress = cur_progress


def df_locs_frames(
        pickle_dir,
        save_path,
):
    files = get_files(pickle_dir)
    count = len(files)
    print(count)
    prev_progress = 0
    res = []
    for i, f in enumerate(files):
        frames = read_pickle(osp.join(pickle_dir, f))
        (
            orientation,
            xangle,
            yangle,
            pants,
            cloth,
            hair,
            action_type,
            label
        ) = filename_to_labels(f)

        res.append(
            (
                frames,
                orientation,
                xangle,
                yangle,
                pants,
                cloth,
                hair,
                action_type,
                label
            )
        )

        cur_progress = int((i + 1) * 100 / count)
        if cur_progress >= prev_progress + 2:
            print(f"progress: {cur_progress}%")
            prev_progress = cur_progress

    df = pd.DataFrame(data=dict(zip(
        [
            "images",
            "orientation",
            "xangle",
            "yangle",
            "pants",
            "cloth",
            "hair",
            "action_type",
            "label"
        ],
        np.transpose(res)
    )))

    df.to_pickle(save_path)

    return df

