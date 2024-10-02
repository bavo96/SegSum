# import pickle
import json
import math
import pickle

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy

dataset_name = "TVSum"
# dataset_name = "SumMe"
data_path = (
    f"./data/{dataset_name}/eccv16_dataset_{dataset_name.lower()}_google_pool5.h5"
)
hdf = h5py.File(data_path, "r")  # Open hdf file
if dataset_name == "SumMe":
    HOMEDATA = "../../data/SumMe/GT/"
elif dataset_name == "TVSum":
    HOMEDATA = "../../data/tvsum50_ver_1_1/ydata-tvsum50-v1_1"

model_score = pickle.load(
    open(f"./analyze/{dataset_name.lower()}_video_scores.pickle", "rb")
)

splits = json.loads(
    open(f"./data/splits/{dataset_name.lower()}_splits.json", "r").read()
)

# SumMe: max method, sigma=0.3, seed=6, blocksize = 1, split 2
# TVSum: max method, sigma=0.9, seed=2, blocksize = 2, split 4

# if dataset_name == "SumMe":
#     s = 2
# elif dataset_name == "TVSum":
#     s = 4

if dataset_name == "SumMe":
    s = 2
elif dataset_name == "TVSum":
    s = 4

# keys = splits[s]["train_keys"]
keys = splits[s]["test_keys"]

print(len(keys))
# print(model_score)


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]


def visualize(video_name):
    # CA-SUM data
    user_summary = np.array(hdf[video_name]["user_summary"])

    # Get video name
    if dataset_name == "SumMe":
        video_full_name = np.array(hdf[video_name]["video_name"]).astype(str).tolist()
    elif dataset_name == "TVSum":
        frame2video = {}
        video2index = {}
        index2video = {}
        for key in list(hdf.keys()):
            nframes = int(np.array(hdf[key]["n_frames"]))
            full_name = np.array(hdf[key]["n_frames"]).astype("str").tolist()
            frame2video[nframes] = key

        for vidname, data in model_score.items():
            video2index[vidname] = frame2video[data["summary"].shape[0]]
            index2video[frame2video[data["summary"].shape[0]]] = vidname
        video_full_name = index2video[video_name]

    n_frames = np.array(hdf[video_name]["n_frames"])

    # l_frame = [i for i in range(n_frames)]
    # l_frame = list(divide_chunks(l_frame, 60))
    # change_points = [[item[0], item[-1]] for item in l_frame]
    change_points = np.array(hdf[video_name]["change_points"])

    if dataset_name == "SumMe":
        # Raw data
        gt_file = HOMEDATA + "/" + video_full_name + ".mat"
        gt_data = scipy.io.loadmat(gt_file)  # dict type
        fps = gt_data["FPS"][0][0]
        fps = math.ceil(fps)
        print("fps:", fps)
        # segments = gt_data["segments"]
        # num_frames = gt_data["nFrames"][0][0]
        # vid_duration = gt_data["video_duration"][0][0]
        # gt_score = gt_data["gt_score"]
        # user_score = gt_data["user_score"]
        # print(f"Raw fps:{fps}")
        # print(f"Round fps: {fps}")
        # print(f"Number of frames: {num_frames}")
        # print(f"Video duration: {vid_duration}")
        # print(f"GT score: {gt_score.shape}")
        # print(f"User score: {user_score.shape}")
    elif dataset_name == "TVSum":
        video_path = f"{HOMEDATA}/ydata-tvsum50-video/video/{video_full_name}.mp4"
        print(video_path)
        vidcap = cv2.VideoCapture(video_path)
        print(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        fps = math.ceil(fps)
        print("fps:", fps)

    # print(segments)
    # print(f"Segments:{segments.shape}")

    kts_segments = []
    len_kts_seg = []

    s = 1
    flag = 1

    # Get change point of KTS
    print(f"Number of changepoint: {len(change_points)}")
    for cp in change_points:
        num_frame = cp[1] - cp[0] + 1
        len_kts_seg.append(num_frame)

        for i in range(num_frame):
            kts_segments.append(s)

        s += flag
        flag *= -1

    # Create result image
    img = np.ones((950, n_frames, 4), np.uint8) * 255

    # img = cv2.putText(
    #     img,
    #     "KTS",
    #     (50, 850),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     2,
    #     (122, 160, 255, 0),
    #     cv2.LINE_AA,
    # )

    # Draw kts
    cv2.rectangle(img, (0, 750), (int(n_frames), 800), (143, 255, 255, 0), -1)

    for i, cp in enumerate(change_points):
        if i == len(change_points) - 1:
            break
        # num_frame = cp[1] - cp[0] + 1
        cv2.line(img, (cp[1], 750), (cp[1], 800), (0, 128, 139, 0), thickness=5)

    # # Draw kts time range
    # for i in range(img.shape[1]):
    #     if i % (fps * 5) == 0:
    #         img = cv2.putText(
    #             img,
    #             str(int(i / fps)),
    #             (i, 970),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             1,
    #             (255, 255, 0),
    #             2,
    #         )

    # # Draw kts
    # for i in range(img.shape[1]):
    #     if kts_segments[i] == 1:
    #         img[900:920, i] = (255, 0, 0)  # Blue
    #     elif kts_segments[i] == 2:
    #         img[900:920, i] = (0, 255, 0)  # Green
    #
    #     if i % (fps * 5) == 0:
    #         img = cv2.putText(
    #             img,
    #             str(int(i / fps)),
    #             (i, 960),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             1,
    #             (255, 255, 0),
    #             2,
    #         )

    # img = cv2.putText(
    #     img,
    #     "Summary",
    #     (50, 1100),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     2,
    #     (122, 160, 255, 0),
    #     cv2.LINE_AA,
    # )

    # Draw model scores
    video_score = model_score[video_full_name]["summary"]
    for i in range(n_frames):
        if video_score[i] == 0:
            img[810:860, i] = (255, 0, 0, 0)  # Blue
        elif video_score[i] == 1:
            img[810:860, i] = (0, 255, 0, 0)  # Green
        else:
            print(f"video_score:{video_score[i]}")
            img[810:860, i] = (0, 255, 255, 0)

    # Draw time range
    for i in range(n_frames):
        if i % (fps * 5) == 0:
            img = cv2.putText(
                img,
                str(int(i / fps)),
                (i, 890),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0, 0),
                2,
            )

    # Draw user's segments
    len_user_seg = []
    num_user = user_summary.shape[0]
    for user in range(user_summary.shape[0]):
        if dataset_name == "SumMe":
            if model_score[f"best_user_{video_full_name}"] == user:
                color = (255, 191, 0, 0)
            else:
                color = (0, 0, 255, 0)
        else:
            color = (0, 0, 255, 0)
        user_sum = [
            i if user_summary[user][i] != 0 else 0
            for i in range(len(user_summary[user]))
        ]

        # Get user segments to do analysis
        user_segment = []
        for i in range(len(user_sum)):
            if (i == 0 and user_sum[i] != 0) or (
                i != 0 and user_sum[i - 1] == 0 and user_sum[i] != 0
            ):
                start = user_sum[i]
            elif (i == len(user_sum) - 1 and user_sum[i] != 0) or (
                i != len(user_sum) - 1 and user_sum[i + 1] == 0 and user_sum[i] != 0
            ):
                end = user_sum[i]
                len_user_seg.append(end - start + 1)
                user_segment.append([start, end])

        # Draw user score
        for i in range(n_frames):
            if user_summary[user][i] == 1:
                img[
                    int(700 / num_user) * (user + 1) : int(700 / num_user) * (user + 2)
                    - 10,
                    i,
                ] = color

    # Draw scores

    res_path = f"./analyze/visualize_data/{dataset_name}/{video_full_name}.jpeg"
    cv2.imwrite(res_path, img)
    return len_kts_seg, len_user_seg


# TVSum: XzYM3PfTM4w
# SumMe: St Maarten Landing

if __name__ == "__main__":
    import os
    import shutil
    from pathlib import Path

    parent_path = f"./analyze/visualize_data/{dataset_name}/"

    if os.path.exists(parent_path) and os.path.isdir(parent_path):
        shutil.rmtree(parent_path)

    Path(parent_path).mkdir(parents=True, exist_ok=True)

    final_len_seg = []
    final_user_len_seg = []
    # for i, key in enumerate(hdf.keys()):
    for i, key in enumerate(keys):
        print(f"video {i}...")
        kts_segments, user_segments = visualize(key)
        final_len_seg.extend(kts_segments)
        final_user_len_seg.extend(user_segments)

    # final_len_seg = sorted(final_len_seg)
    # (n, bins, patches) = plt.hist(final_len_seg, bins=40, label="hst")
    # print(n)
    # print(bins)
    # print(patches)
    # n_np, bins_np = np.histogram(final_len_seg, 10)
    # print(n_np, bins_np)
    # for item in final_len_seg:
    #     if item >= 7 and item <= 219.6:
    #         print(item, end=" ")

    final_user_len_seg = sorted(final_user_len_seg)
    (n, bins, patches) = plt.hist(final_user_len_seg, bins=40, label="hst")
    n_np, bins_np = np.histogram(final_len_seg, 10)
    # print("a")
    # print(n)
    # print("a")
    # print(bins)
    # print("a")
    # print(patches)
    # print("a")
    # print(n_np, bins_np)
    # print("a")
    # plt.savefig("./visualize/hist.jpeg")

# if __name__ == "__main__":
#     user_sum = [1, 2, 3, 4, 0, 0, 0, 0, 9, 10, 11, 12]
#
#     # def divide_chunks(l, n):
#     #     # looping till length l
#     #     for i in range(0, len(l), n):
#     #         yield l[i : i + n]
#     #
#     # res = list(divide_chunks(a, 2))
#     # print(res)
#     user_segment = []
#     for i in range(len(user_sum)):
#         if (i == 0 and user_sum[i] != 0) or (
#             i != 0 and user_sum[i - 1] == 0 and user_sum[i] != 0
#         ):
#             start = user_sum[i]
#         elif (i == len(user_sum) - 1 and user_sum[i] != 0) or (
#             i != len(user_sum) - 1 and user_sum[i + 1] == 0 and user_sum[i] != 0
#         ):
#             end = user_sum[i]
#             user_segment.append([start, end])
#
#     print(user_segment)
