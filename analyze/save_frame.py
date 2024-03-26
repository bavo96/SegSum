import json
import pickle
import sys

import h5py
import numpy as np

np.set_printoptions(threshold=sys.maxsize)
import os
from pathlib import Path

import cv2
import matplotlib.pylab as plt
import seaborn as sns

# dataset_name = "SumMe"
dataset_name = "TVSum"

model_score = pickle.load(open(f"./{dataset_name.lower()}_video_scores.pickle", "rb"))

if dataset_name == "SumMe":
    video_path = "../../data/SumMe/videos/"
elif dataset_name == "TVSum":
    video_path = (
        "../../data/tvsum50_ver_1_1/ydata-tvsum50-v1_1/ydata-tvsum50-video/video/"
    )

data_path = (
    f"../data/{dataset_name}/eccv16_dataset_{dataset_name.lower()}_google_pool5.h5"
)

hdf = h5py.File(data_path, "r")  # Open hdf file


# TVSum: XzYM3PfTM4w
# SumMe: St Maarten Landing

splits = json.loads(
    open(f"../data/splits/{dataset_name.lower()}_splits.json", "r").read()
)
if dataset_name == "SumMe":
    s = 2
elif dataset_name == "TVSum":
    s = 4

test_keys = splits[s]["test_keys"]


# Get video name

video2index = {}
index2video = {}
frame2video = {}

for i, video_name in enumerate(test_keys):
    if dataset_name == "SumMe":
        video_full_name = np.array(hdf[video_name]["video_name"]).astype(str).tolist()
        video2index[video_full_name] = video_name
        index2video[video_name] = video_full_name
    elif dataset_name == "TVSum":
        for key in list(hdf.keys()):
            nframes = int(np.array(hdf[key]["n_frames"]))
            frame2video[nframes] = key

        for video_full_name, data in model_score.items():
            video2index[video_full_name] = frame2video[data["summary"].shape[0]]
            index2video[frame2video[data["summary"].shape[0]]] = video_full_name


for video_name, data in model_score.items():
    if dataset_name == "SumMe":
        if "St Maarten Landing" not in video_name:
            continue
    elif dataset_name == "TVSum":
        if "XzYM3PfTM4w" not in video_name:
            continue
    print("video:", video_name)
    summary = data["summary"]
    attn_weights = data["attn_weights"].cpu()
    if dataset_name == "SumMe":
        best_user = data["best_user"]

    unique, counts = np.unique(summary, return_counts=True)
    values = dict(zip(unique, counts))
    print(values[1] / len(summary))

    print(attn_weights.shape)

    index = video2index[video_name]
    n_frames = np.array(hdf[index]["n_frames"])
    change_points = np.array(hdf[index]["change_points"])
    user_summary = np.array(hdf[index]["user_summary"])
    print(video_name, index)
    print(n_frames)
    print(change_points.shape)
    print(user_summary.shape)
    print(attn_weights.shape[0])
    for i in range(attn_weights.shape[0]):
        print(attn_weights[i][i])

    ax = sns.heatmap(attn_weights, linewidth=0.5)

    idx = 90
    step = 30
    accepted_index = (
        [i for i in range(0, idx, step)]
        + [i for i in range(int(n_frames / 2), int(n_frames / 2) + idx, step)]
        + [i for i in range(n_frames - idx, n_frames, step)]
    )

    num_imgs = 12
    step = int(n_frames / num_imgs)
    print(step)

    accepted_index = [i for i in range(0, n_frames, step)]

    print(accepted_index)

    count = 0

    parent_path = f"frames/{video_name}/"
    Path(parent_path).mkdir(parents=True, exist_ok=True)

    video_full_path = os.path.join(video_path, f"{video_name}.mp4")
    print(video_full_path)
    video = cv2.VideoCapture(video_full_path)
    print(video)
    print("opened!")
    while True:
        success, image = video.read()
        print(success, image)
        print("read!")
        if success:
            print("success")
            if count in accepted_index:
                print("save")
                cv2.imwrite(parent_path + video_name + str(count) + ".jpg", image)
        else:
            print("failed!")
            break
        count += 1

        if video.get(cv2.CAP_PROP_POS_FRAMES) == video.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the end of the video has been reached, break out of the loop
            break

    video.release()
    cv2.destroyAllWindows()
    print("closed!")
