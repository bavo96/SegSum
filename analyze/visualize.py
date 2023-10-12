# import pickle
import math

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy

data_path = "../../data/formatted_data/SumMe/eccv16_dataset_summe_google_pool5.h5"
hdf = h5py.File(data_path, "r")  # Open hdf file
HOMEDATA = "../../data/SumMe/GT/"


def raw_video(video_name):
    # CA-SUM data
    user_summary = np.array(hdf[video_name]["user_summary"])
    video_full_name = np.array(hdf[video_name]["video_name"]).astype(str).tolist()
    change_points = np.array(hdf[video_name]["change_points"])

    # Raw data
    gt_file = HOMEDATA + "/" + video_full_name + ".mat"
    gt_data = scipy.io.loadmat(gt_file)  # dict type
    fps = gt_data["FPS"][0][0]
    fps = math.ceil(fps)
    # segments = gt_data["segments"]
    num_frames = gt_data["nFrames"][0][0]
    vid_duration = gt_data["video_duration"][0][0]
    gt_score = gt_data["gt_score"]
    user_score = gt_data["user_score"]

    # print(segments)
    # print(f"Segments:{segments.shape}")

    print(f"Raw fps:{fps}")
    print(f"Round fps: {fps}")
    print(f"Number of frames: {num_frames}")
    print(f"Video duration: {vid_duration}")
    print(f"GT score: {gt_score.shape}")
    print(f"User score: {user_score.shape}")

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

    img = np.zeros((1024, len(kts_segments), 3), np.uint8)

    # Draw kts
    for i in range(img.shape[1]):
        if kts_segments[i] == 1:
            img[800:820, i] = (255, 0, 0)
        elif kts_segments[i] == 2:
            img[800:820, i] = (0, 255, 0)

        if i % (fps * 5) == 0:
            img = cv2.putText(
                img,
                str(int(i / fps)),
                (i, 860),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
            )

    # Draw user's segments
    len_user_seg = []
    num_user = user_summary.shape[0]
    for user in range(user_summary.shape[0]):
        user_sum = [
            i if user_summary[user][i] != 0 else 0
            for i in range(len(user_summary[user]))
        ]

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

        for i in range(img.shape[1]):
            if user_summary[user][i] == 1:
                img[
                    int(700 / num_user) * (user + 1) : int(700 / num_user) * (user + 2)
                    - 10,
                    i,
                ] = (0, 0, 255)

    cv2.imwrite(f"./visualize/{video_full_name}.jpeg", img)
    return len_kts_seg, len_user_seg


if __name__ == "__main__":
    final_len_seg = []
    final_user_len_seg = []
    for i, key in enumerate(hdf.keys()):
        print(f"video {i}...")
        kts_segments, user_segments = raw_video(key)
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
    print("a")
    print(n)
    print("a")
    print(bins)
    print("a")
    print(patches)
    print("a")
    n_np, bins_np = np.histogram(final_len_seg, 10)
    print(n_np, bins_np)
    print("a")
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
