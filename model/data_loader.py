# -*- coding: utf-8 -*-
import json
import pickle

import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch import Tensor

from model.config import get_config


class VideoData(Dataset):
    def __init__(self, config, raw_video_features, hdf, splits):
        self.dataset_name = config.dataset_name
        self.mode = config.mode
        self.split_index = config.split_index
        self.video2index = {}
        self.index2video = {}
        self.all_data = []
        self.change_points = []

        # Dataset
        self.hdf = hdf
        self.splits = splits
        self.raw_video_features = raw_video_features
        self.keys = splits[self.split_index][self.mode + "_keys"]

        ## Get name mapping
        frame2video = {}
        if config.dataset_name == "SumMe":
            for key in list(self.hdf.keys()):
                video_fullname = (
                    np.array(self.hdf.get(key + "/video_name")).astype("str").tolist()
                )
                self.video2index[video_fullname] = key
                self.index2video[key] = video_fullname

        elif config.dataset_name == "TVSum":
            for key in list(self.hdf.keys()):
                nframes = int(np.array(self.hdf[key]["n_frames"]))
                frame2video[nframes] = key

            for k, v in self.raw_video_features.items():
                video_fullname = k
                self.video2index[video_fullname] = frame2video[v.shape[0]]
                self.index2video[frame2video[v.shape[0]]] = video_fullname

        for video_name in self.keys:
            # print(hdf[video_name].keys())
            # print(hdf[video_name]["n_frames"])
            # print(video_name)
            # print(self.hdf[video_name].keys())
            change_point = np.array(self.hdf[video_name]["change_points"])
            self.change_points.append(change_point)
            full_vid_name = self.index2video[video_name]
            # print(full_vid_name)
            video_features = self.raw_video_features[full_vid_name]
            self.all_data.append((video_name, video_features, change_point))

        # # Get subsampled features
        # for video_name in splits[self.split_index][self.mode + "_keys"]:
        #     # print(hdf[video_name].keys())
        #     # print(hdf[video_name]["n_frame_per_seg"])
        #     # print(np.array(hdf[video_name]["change_points"]))
        #     video_features = Tensor(np.array(hdf[video_name + "/features"]))
        #     self.all_data.append((video_name, video_features))

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        video_name, feature, change_point = self.all_data[index]
        return video_name, feature, change_point
        # video_name, feature = self.all_data[index]
        # return video_name, feature


def get_loader(
    config,
    raw_video_features,
    hdf,
    splits,
):
    video_dataset = VideoData(config, raw_video_features, hdf, splits)
    if config.mode.lower() == "train":
        return DataLoader(video_dataset, batch_size=1, shuffle=True)
    elif config.mode.lower() == "test":
        return video_dataset


if __name__ == "__main__":
    # DATASET NAME
    dataset_name = "SumMe"
    # dataset_name = 'TVSum'

    # Load default config
    config = get_config(dataset_name=dataset_name)
    print(config)

    # LOAD MANDATORY DATA
    # Load features
    print(f"Training on {dataset_name} dataset...")
    print(f"Get features of original video...")
    print(f"Features path: {config.features_path}")
    raw_video_features = pickle.load(open(config.features_path, "rb"))
    # Load dataset
    hdf = h5py.File(config.data_path, "r")  # Open hdf file
    # Load splits.json
    splits = json.loads(open(config.split_path, "r").read())

    config.mode = "train"
    data_loader = get_loader(config, raw_video_features, hdf, splits)
    # for data in data_loader:
    #     print(data)
    print("Number of train videos:", len(data_loader))

    config.mode = "test"
    data_loader = get_loader(config, raw_video_features, hdf, splits)

    # for data in data_loader:
    #     print(data)
    print("Number of test videos:", len(data_loader))
