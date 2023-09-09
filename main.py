# -*- coding: utf-8 -*-
import numpy as np
from config import get_config
# from data_loader import get_loader
from data_loader_new import get_loader
# from solver import Solver
from solver_new import Solver

if __name__ == "__main__":
    ## Define paths
    split_path = "./data/splits/summe_splits.json"
    data_path = "./data/SumMe/eccv16_dataset_summe_google_pool5.h5"
    root_path = "./data/SumMe/videos/"
    raw_data_path = "./data/video_features.pickle"

    ## Old model
    # train_config = get_config(mode="train")
    # test_config = get_config(mode="test")
    # train_loader = get_loader(
    #     train_config.mode, train_config.video_type, train_config.split_index
    # )
    # test_loader = get_loader(
    #     test_config.mode, test_config.video_type, test_config.split_index
    # )

    result = []
    res_min_loss = []

    # best score 53.40%: seed=3000, no attention (no block_size)
    # best score 56.71%: seed=3000, with attention (no block_size)

    for split_idx in range(5):
        print(f"Current split index: {split_idx}")
        ## New model
        train_config = get_config(
            mode="train",
            reg_factor=0.15,
            n_epochs=120,
            seed=3000,
            block_size=2,
            split_index=split_idx,
        )
        # test_config = get_config(mode="test", reg_factor=0.15)
        train_loader = get_loader(
            root_path, split_path, data_path, "train", split_idx, raw_data_path
        )
        test_loader = get_loader(
            root_path, split_path, data_path, "test", split_idx, raw_data_path
        )

        solver = Solver(train_config, train_loader, test_loader)

        solver.build()

        # solver.evaluate(
        #     -1
        # )  # evaluates the summaries using the initial random weights of the network
        max_epoch, max_fscore, min_epoch, min_loss, min_fscore = solver.train()

        print(f"Epoch with highest score: epoch {max_epoch} ({max_fscore:.2f}%)")
        print(f"Epoch with lowest loss: epoch {min_epoch} ({min_fscore:.2f}%)")

        result.append((max_epoch, max_fscore))
        res_min_loss.append((min_epoch, min_loss, min_fscore))

print("Choosing best fscore...")
l_fscore = []
for i, (epoch, fscore) in enumerate(result):
    print(f"Result for split {i}: epoch={epoch}, fscore={fscore:.2f}%")
    l_fscore.append(fscore)

print(f"Average fscore: {np.mean(l_fscore):.2f}%")

print("Choosing best loss...")
l_fscore = []
for i, (epoch, loss, fscore) in enumerate(res_min_loss):
    print(f"Result for split {i}: epoch={epoch}, loss={loss}, fscore={fscore:.2f}%")
    l_fscore.append(fscore)

print(f"Average fscore: {np.mean(l_fscore):.2f}%")
