import pickle

import h5py
import numpy as np

dataset_name = "TVSum"
raw_data_path = (
    f"./data/{dataset_name}/{dataset_name.lower()}_video_features_version_2.pickle"
)
data_path = f"./data/{dataset_name}/eccv16_dataset_{dataset_name.lower()}_google_pool5.h5"  ## Get raw features

frames = {}

print(f"Get features of original video...")
raw_video_features = pickle.load(
    open(raw_data_path, "rb"),
)
for k, v in raw_video_features.items():
    frames[k] = len(v)
    print(k, v.shape)


print()
print(f"Get video info...")
hdf = h5py.File(data_path, "r")  # Open hdf file
min_cp = 100000
for key in hdf:
    user_summary = np.array(hdf.get(key + "/user_summary")).astype("str").tolist()
    gt_score = (
        np.array(hdf.get(key + "/gtscore")).astype("str").tolist()
    )  # Inference CA-SUM with sub-sampled
    n_frames = np.array(hdf.get(key + "/n_frames")).astype("str").tolist()
    cp = np.array(hdf.get(key + "/change_points"))

    print(f"key: {key}")
    print(len(user_summary[0]))
    print(len(gt_score))
    print(n_frames)
    print(f"change point shape: {cp.shape}")
    print(f"number of changepoints: {cp.shape[0]}")
    # for p in cp:
    #     print(p)
    if cp.shape[0] < min_cp:
        min_cp = cp.shape[0]

print("min cp:", min_cp)


# for key in hdf:
#     video_fullname = np.array(hdf.get(key + "/video_name")).astype("str").tolist()
#     video_frame = np.array(hdf.get(key + "/n_frames"))
#     print(
#         f"video name: {video_fullname}, features frame: {frames[video_fullname]}, gt frames: {video_frame}, correct data: {video_frame==frames[video_fullname]}"
#     )
