# -*- coding: utf-8 -*-
import json
import pickle

import h5py
import numpy as np

from model.config import get_config
from model.data_loader import get_loader
from model.solver import Solver

if __name__ == "__main__":
    # DATASET NAME
    dataset_name = "SumMe"
    # dataset_name = "TVSum"

    config = get_config(dataset_name=dataset_name, root_data_path="./data")

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

    seed_scores = []
    # seeds = range(16, 17)
    seeds = [16]

    for seed in seeds:
        print(f"Random seed {seed}/3000...")
        # Save scores on test set
        result = []
        res_min_loss = []
        for split_idx in range(5):
            print(f"Current split index: {split_idx}")

            # Set params
            params = {
                "reg_factor": 0.15,
                "n_epochs": 200,
                "seed": seed,
                # seed=1996,
                # seed=3407,
                "block_size": 1,
                "split_index": split_idx,
            }
            for k, v in params.items():
                setattr(config, k, v)
            # print(config)

            # print(f"Get test loader...")
            config.mode = "test"
            test_loader = get_loader(config, raw_video_features, hdf, splits)
            # print(f"Get train loader...")
            config.mode = "train"
            train_loader = get_loader(config, raw_video_features, hdf, splits)
            # print(f"Create solver...")
            solver = Solver(config, train_loader, test_loader, splits)

            solver.build()

            # solver.evaluate(
            #     -1
            # )  # evaluates the summaries using the initial random weights of the network

            # max_epoch, max_fscore, min_epoch, min_loss, min_fscore = solver.train()
            max_epoch, max_precision, max_recall, max_fscore = solver.train()

            # print(f"Epoch with highest score: epoch {max_epoch} ({max_fscore:.2f}%)")
            # print(f"Epoch with lowest loss: epoch {min_epoch} ({min_fscore:.2f}%)")

            result.append((max_epoch, max_precision, max_recall, max_fscore))
            # res_min_loss.append((min_epoch, min_loss, min_fscore))

        print("Choosing best fscore...")
        l_score = []
        for i, (epoch, precision, recall, fscore) in enumerate(result):
            print(
                f"Result for split {i}: epoch={epoch}, precision={precision:.2f}%, recall={recall:.2f}%, fscore={fscore:.2f}%"
            )
            l_score.append([precision, recall, fscore])

        l_score = np.mean(l_score, axis=0)

        print(
            f"Average score: precision: {l_score[0]:.2f}%, recall: {l_score[1]:.2f}%, fscore: {l_score[2]:.2f}%"
        )
        seed_scores.append(l_score)

    mean_score = np.mean(seed_scores, axis=0)
    print(
        f"Average score after 3000 seeds: precision: {mean_score[0]:.2f}%, recall: {mean_score[1]:.2f}%, fscore: {mean_score[2]:.2f}%"
    )

    # print("Choosing best loss...")
    # l_fscore = []
    # for i, (epoch, loss, fscore) in enumerate(res_min_loss):
    #     print(f"Result for split {i}: epoch={epoch}, loss={loss}, fscore={fscore:.2f}%")
    #     l_fscore.append(fscore)
    #
    # print(f"Average fscore: {np.mean(l_fscore):.2f}%")
