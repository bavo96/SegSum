# -*- coding: utf-8 -*-
# Code to train a single dataset
import json
import os
import pickle
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd

from model.config import get_config
from model.data_loader import get_loader
from model.solver import Solver

if __name__ == "__main__":
    # DATASET NAME
    # dataset_name = "SumMe"
    dataset_name = "TVSum"

    # EXPERIMENT NAME

    config = get_config(dataset_name=dataset_name, root_data_path="./data")

    # LOAD MANDATORY DATA
    print(f"Training on {dataset_name} dataset...")
    print(f"Prepare data for training...")
    ## Load features
    raw_video_features = pickle.load(open(config.features_path, "rb"))
    ## Load dataset
    hdf = h5py.File(config.data_path, "r")  # Open hdf file
    ## Load splits.json
    splits = json.loads(open(config.split_path, "r").read())

    # SumMe: max method, sigma=0.3, seed=6, blocksize = 1, split 2
    # TVSum: max method, sigma=0.9, seed=2, blocksize = 2, split 4

    # PARAMS AND RESULTS
    ## Experiment document
    n_epoch = 400
    if dataset_name == "SumMe":
        batch_size = 20
        reg_factor = 0.3
        block_size = 1
        seed = 6
        list_split_idx = [2]
    elif dataset_name == "TVSum":
        batch_size = 40
        reg_factor = 0.9
        block_size = 2
        seed = 2
        list_split_idx = [4]
    input_size = 1024
    seg_emb_methods = "max"  # "max", "mean", "attention"
    attn_mechanism = True

    result = []  # Save scores on test set
    for split_idx in list_split_idx:
        print(f"Current split index: {split_idx}")
        # Set params
        params = {
            "seed": seed,
            "reg_factor": reg_factor,
            "n_epochs": n_epoch,
            "block_size": block_size,
            "split_index": split_idx,
            "batch_size": batch_size,
            "input_size": input_size,
            "seg_emb_method": seg_emb_methods,
            "attn_mechanism": attn_mechanism,
        }
        for k, v in params.items():
            setattr(config, k, v)
        print(config)
        config.set_training_dir(config.seed, config.reg_factor, config.dataset_name)

        config.mode = "test"
        test_loader = get_loader(config, raw_video_features, hdf, splits)
        config.mode = "train"
        train_loader = get_loader(config, raw_video_features, hdf, splits)
        solver = Solver(config, train_loader, test_loader, splits)

        solver.build()

        (
            max_epoch,
            max_precision,
            max_recall,
            max_fscore,
        ) = solver.train()

        result.append((max_epoch, max_precision, max_recall, max_fscore))

    print("Print result...")
    l_score = []
    for i, (epoch, precision, recall, fscore) in enumerate(result):
        print(
            f"Result for split {i}: epoch={epoch}, precision={precision:.2f}%, recall={recall:.2f}%, fscore={fscore:.2f}%"
        )
        l_score.append([precision, recall, fscore])

    mean_score = np.mean(l_score, axis=0)

    print(
        f"Average score for splits: precision={mean_score[0]:.2f}%, recall={mean_score[1]:.2f}%, fscore={mean_score[2]:.2f}%"
    )
