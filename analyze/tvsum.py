import pickle

import h5py
import numpy as np

raw_data_path = "tvsum_video_features.pickle"
data_path = "eccv16_dataset_tvsum_google_pool5.h5"  ## Get raw features

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
for key in hdf:
    user_summary = np.array(hdf.get(key + "/user_summary")).astype("str").tolist()
    gt_score = (
        np.array(hdf.get(key + "/gtscore")).astype("str").tolist()
    )  # Inference CA-SUM with sub-sampled
    n_frames = np.array(hdf.get(key + "/n_frames")).astype("str").tolist()

    print(f"key: {key}")
    print(len(user_summary[0]))
    print(len(gt_score))
    print(n_frames)

# for key in hdf:
#     video_fullname = np.array(hdf.get(key + "/video_name")).astype("str").tolist()
#     video_frame = np.array(hdf.get(key + "/n_frames"))
#     print(
#         f"video name: {video_fullname}, features frame: {frames[video_fullname]}, gt frames: {video_frame}, correct data: {video_frame==frames[video_fullname]}"
#     )
