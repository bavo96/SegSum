# -*- coding: utf-8 -*-

import json
import os
import pickle
import random

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class Rescale(object):
    """Rescale a image to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, *output_size):
        self.output_size = output_size

    def __call__(self, image):
        """
        Args:
            image (PIL.Image) : PIL.Image object to rescale
        """
        new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = image.resize((new_w, new_h), resample=Image.BILINEAR)
        return img


resnet_transform = transforms.Compose(
    [
        Rescale(224, 224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


class VideoData(Dataset):
    def __init__(
        self,
        root_path,
        split_path,
        data_path,
        mode="train",
        split=0,
        raw_data_path="",
    ):
        self.root_path = root_path
        self.mode = mode
        self.split_index = split
        self.video2index = {}
        self.index2video = {}
        self.list_video_features = []
        self.raw_data_path = raw_data_path
        self.list_change_point = []

        self.hdf = h5py.File(data_path, "r")  # Open hdf file

        print(f"Mode: {self.mode}")

        with open(split_path, "r") as f:
            splits = json.loads(f.read())
            self.keys = splits[self.split_index][f"{self.mode}_keys"]

        ## Get name mapping
        for key in list(self.hdf.keys()):
            video_fullname = (
                np.array(self.hdf.get(key + "/video_name")).astype("str").tolist()
            )
            self.video2index[video_fullname] = key
            self.index2video[key] = video_fullname

        ## Get raw features
        self.raw_video_features = pickle.load(
            open(self.raw_data_path, "rb"),
        )
        for video_name in splits[self.split_index][self.mode + "_keys"]:
            # print(hdf[video_name].keys())
            # print(hdf[video_name]["n_frames"])
            change_point = np.array(self.hdf[video_name]["change_points"])
            self.list_change_point.append(change_point)
            full_vid_name = self.index2video[video_name]

            video_features = torch.squeeze(
                torch.stack(self.raw_video_features[full_vid_name])
            )
            # print(video_features.shape)
            self.list_video_features.append((video_name, video_features, change_point))

        ## Get subsampled features
        # for video_name in splits[self.split_index][self.mode + "_keys"]:
        #     print(hdf[video_name].keys())
        #     print(hdf[video_name]["n_frame_per_seg"])
        #     print(np.array(hdf[video_name]["change_points"]))
        #     video_features = torch.Tensor(np.array(hdf[video_name + "/features"]))
        #     self.list_video_features.append((video_name, video_features))

        ## Close hdf

    def __len__(self):
        return len(self.list_video_features)

    def __getitem__(self, index):
        video_name, feature, change_point = self.list_video_features[index]
        return video_name, feature, change_point


def get_loader(root_path, split_path, data_path, mode, split, raw_data_path):
    if mode.lower() == "train":
        video_dataset = VideoData(
            root_path,
            split_path,
            data_path,
            mode=mode.lower(),
            split=split,
            raw_data_path=raw_data_path,
        )
        return DataLoader(video_dataset, batch_size=1, shuffle=True)
    elif mode.lower() == "test":
        video_dataset = VideoData(
            root_path,
            split_path,
            data_path,
            mode=mode.lower(),
            split=split,
            raw_data_path=raw_data_path,
        )
        return video_dataset


if __name__ == "__main__":
    # data_loader = get_loader("SumMe", "train")
    # for batch in data_loader:
    #     print(batch)
    #     break
    split_path = "../../data/formatted_data/splits/summe_splits.json"
    data_path = "../../data/formatted_data/SumMe/eccv16_dataset_summe_google_pool5.h5"
    root_path = "../../data/SumMe/videos/"
    raw_data_path = "./data/video_features.pickle"

    # data_loader = get_loader(
    #     root_path, split_path, data_path, "train", 0, raw_data_path
    # )
    # print("Number of train videos:", len(data_loader))
    data_loader = get_loader(root_path, split_path, data_path, "test", 0, raw_data_path)
    for data in data_loader:
        print(data)
        print(len(data[2]))
    print("Number of test videos:", len(data_loader))
