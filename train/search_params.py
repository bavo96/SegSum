# -*- coding: utf-8 -*-
import json
import pickle
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd

from model.config import get_config
from model.data_loader import get_loader
from model.solver import Solver

if __name__ == "__main__":
    # EXPERIMENT NAME
    experiment_id = "1"

    # DATASET NAME
    # dataset_name = "SumMe"
    dataset_name = "TVSum"

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

    # PARAMS AND RESULTS
    ## Experiment document
    info = defaultdict(list)
    seed_scores = []
    seeds = range(1, 11)
    # seeds = [1996]  # SumMe
    # seeds = [3407]  # SumMe
    # seeds = [3000]  # SumMe
    # seeds = [16]  # SumMe
    block_size = 1
    n_epoch = 400
    batch_size = 20

    for seed in seeds:
        print(f"Random seed {seed}...")
        # Save scores on test set
        result = []
        res_min_loss = []
        for split_idx in range(5):
            print(f"Current split index: {split_idx}")

            # Set params
            params = {
                "reg_factor": 0.15,
                "n_epochs": n_epoch,
                "seed": seed,
                # seed=1996,
                # seed=3407,
                "block_size": block_size,
                "split_index": split_idx,
                "batch_size": 20,
            }
            for k, v in params.items():
                setattr(config, k, v)
            print(config)
            config.set_training_dir(config.seed, config.reg_factor, config.dataset_name)

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

        print("Print result...")
        l_score = []
        for i, (epoch, precision, recall, fscore) in enumerate(result):
            print(
                f"Result for split {i}: epoch={epoch}, precision={precision:.2f}%, recall={recall:.2f}%, fscore={fscore:.2f}%"
            )
            l_score.append([precision, recall, fscore])

        mean_score = np.mean(l_score, axis=0)

        print(
            f"Average score: precision={mean_score[0]:.2f}%, recall={mean_score[1]:.2f}%, fscore={mean_score[2]:.2f}%"
        )
        seed_scores.append(mean_score)
        info["seed"].append(seed)
        info["block_size_attention_matrix"].append(block_size)
        info["batch_size"].append(batch_size)
        info["n_epoch"].append(n_epoch)
        info["splits_max_fscore"].append(np.max(l_score, axis=0)[2])
        info["splits_min_fscore"].append(np.min(l_score, axis=0)[2])
        info["splits_avg_precision"].append(mean_score[0])
        info["splits_avg_recall"].append(mean_score[1])
        info["splits_avg_fscore"].append(mean_score[2])

    mean_seed_score = np.mean(seed_scores, axis=0)
    print(
        f"Average score after 3000 seeds: precision={mean_seed_score[0]:.2f}%, recall={mean_seed_score[1]:.2f}%, fscore={mean_seed_score[2]:.2f}%"
    )
    df = pd.DataFrame(info)

    df.loc[len(df)] = [
        "",
        "",
        "",
        "",
        np.mean(info["splits_max_fscore"]),
        np.mean(info["splits_min_fscore"]),
        np.mean(info["splits_avg_precision"]),
        np.mean(info["splits_avg_recall"]),
        np.mean(info["splits_avg_fscore"]),
    ]

    # df.to_csv("", index=False)
    df.to_excel(
        f"./results/experiment_{experiment_id}_{dataset_name}.xlsx",
        sheet_name="result",
        index=False,
    )
    # print("Choosing best loss...")
    # l_fscore = []
    # for i, (epoch, loss, fscore) in enumerate(res_min_loss):
    #     print(f"Result for split {i}: epoch={epoch}, loss={loss}, fscore={fscore:.2f}%")
    #     l_fscore.append(fscore)
    #
    # print(f"Average fscore: {np.mean(l_fscore):.2f}%")
