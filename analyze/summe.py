import pickle

import h5py
import numpy as np
import torch

raw_data_path = "./data/summe_video_features.pickle"
# raw_data_path = "./data/SumMe/summe_gt_video_features.pickle"
# raw_data_path = "./data/SumMe/video_features.pickle"
data_path = "eccv16_dataset_summe_google_pool5.h5"  ## Get raw features

frames = {}

print(f"Get features of original video...")
raw_video_features = pickle.load(
    open(raw_data_path, "rb"),
)

video_features = {}

for video_name, batch_feats in raw_video_features.items():
    # batch_feats = torch.cat(list_feats, dim=0).squeeze()
    print(video_name, batch_feats.shape)
    # video_features[video_name] = batch_feats

# print()
# print(f"Get video info...")
# hdf = h5py.File(data_path, "r")  # Open hdf file
#
# for key in hdf:
#     video_fullname = np.array(hdf.get(key + "/video_name")).astype("str").tolist()
#     video_frame = np.array(hdf.get(key + "/n_frames"))
#     print(
#         f"video name: {video_fullname}, features frame: {frames[video_fullname]}, gt frames: {video_frame}, correct data: {video_frame==frames[video_fullname]}"
#     )

# # # Save data to pickle
# with open("./formatted_video_features.pickle", "wb") as handle:
#     pickle.dump(video_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
