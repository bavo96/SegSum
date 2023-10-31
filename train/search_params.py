# -*- coding: utf-8 -*-
import os
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
    # DATASET NAME
    # dataset_name = "SumMe"
    dataset_name = "TVSum"

    # EXPERIMENT NAME
    # experiment_meaning = "seg_emb_method"
    # experiment_meaning = "sigma"
    experiment_meaning = "blocksize"
    excel_path = f"./results/experiment_{experiment_meaning}_{dataset_name}.xlsx"

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
    seed_scores = []
    seeds = range(1, 11)
    n_epoch = 400
    if dataset_name == "SumMe":
        batch_size = 20
    elif dataset_name == "TVSum":
        batch_size = 40
    # round number affects the range, SumMe = [0.2, 0.5], TVSum = [0.7, 0.9]
    # reg_factor = np.arange(0, 1.1, 0.1)  # Compute sigma
    # reg_factor = np.arange(0.2, 0.6, 0.1)  # SumMe
    reg_factor = np.arange(0.7, 1, 0.1)  # TVSum
    reg_factor = np.round(reg_factor, 1)
    # reg_factor = [0.5]
    input_size = 1024
    seg_emb_methods = ["max"]  # "max", "mean", "attention"
    attn_mechanism = True
    block_size = range(1, 6)

    for bsize in block_size:
        for sigma in reg_factor:
            for seg_method in seg_emb_methods:
                info = defaultdict(list)
                for seed in seeds:
                    print(f"Random seed {seed}...")
                    result = []  # Save scores on test set
                    # res_min_loss = []
                    for split_idx in range(5):
                        print(f"Current split index: {split_idx}")
                        # Set params
                        params = {
                            "reg_factor": sigma,
                            "n_epochs": n_epoch,
                            "seed": seed,
                            "block_size": bsize,
                            "split_index": split_idx,
                            "batch_size": batch_size,
                            "input_size": input_size,
                            "seg_emb_method": seg_method,
                            "attn_mechanism": attn_mechanism,
                            "experiment_id": f"{experiment_meaning}_{seg_method}_{sigma}_{bsize}",
                        }
                        for k, v in params.items():
                            setattr(config, k, v)
                        print(config)
                        config.set_training_dir(
                            config.seed, config.reg_factor, config.dataset_name
                        )

                        config.mode = "test"
                        test_loader = get_loader(
                            config, raw_video_features, hdf, splits
                        )
                        config.mode = "train"
                        train_loader = get_loader(
                            config, raw_video_features, hdf, splits
                        )
                        solver = Solver(config, train_loader, test_loader, splits)

                        solver.build()

                        # max_epoch, max_fscore, min_epoch, min_loss, min_fscore = solver.train()
                        (
                            max_epoch,
                            max_precision,
                            max_recall,
                            max_fscore,
                        ) = solver.train()

                        result.append(
                            (max_epoch, max_precision, max_recall, max_fscore)
                        )
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
                        f"Average score for splits: precision={mean_score[0]:.2f}%, recall={mean_score[1]:.2f}%, fscore={mean_score[2]:.2f}%"
                    )
                    seed_scores.append(mean_score)
                    info["seed"].append(seed)
                    info["segment_embedding_method"].append(seg_method)
                    info["regularization factor (sigma)"].append(sigma)
                    info["block_size_attention_matrix"].append(bsize)
                    info["batch_size"].append(batch_size)
                    info["n_epoch"].append(n_epoch)
                    info["splits_max_fscore"].append(np.max(l_score, axis=0)[2])
                    info["splits_min_fscore"].append(np.min(l_score, axis=0)[2])
                    info["splits_avg_precision"].append(mean_score[0])
                    info["splits_avg_recall"].append(mean_score[1])
                    info["splits_avg_fscore"].append(mean_score[2])

                mean_seed_score = np.mean(seed_scores, axis=0)
                print(
                    f"Average score after {len(seeds)} seeds: precision={mean_seed_score[0]:.2f}%, recall={mean_seed_score[1]:.2f}%, fscore={mean_seed_score[2]:.2f}%"
                )
                df = pd.DataFrame(info)

                # Compute mean per seed
                df.loc[len(df)] = [
                    "",
                    "",
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

                if os.path.isfile(excel_path):
                    with pd.ExcelWriter(excel_path, mode="a") as writer:
                        df.to_excel(writer, sheet_name=f"{bsize}_{sigma}", index=False)
                else:
                    with pd.ExcelWriter(excel_path) as writer:
                        df.to_excel(writer, sheet_name=f"{bsize}_{sigma}", index=False)
