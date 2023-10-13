# -*- coding: utf-8 -*-
import json
import os
import random

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm, trange

from inference.inference_data import inference
# from model.summarizer import Sum
from model.new_sum import Sum
from model.utils import TensorboardWriter


class Solver(object):
    def __init__(self, config=None, train_loader=None, test_loader=None, splits=None):
        """Class that Builds, Trains and Evaluates CA-SUM model."""
        # Initialize variables to None, to be safe
        self.model, self.optimizer, self.writer = None, None, None

        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.splits = splits

        # Set the seed for generating reproducible random numbers
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)

    def build(self):
        """Function for constructing the CA-SUM model, its key modules and parameters."""
        # Model creation
        self.model = Sum(
            input_size=2048,
            output_size=2048,
            block_size=self.config.block_size,
        ).to(self.config.device)

        if self.config.init_type is not None:
            self.init_weights(
                net=self.model,
                init_type=self.config.init_type,
                init_gain=self.config.init_gain,
            )

        if self.config.mode == "train":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.l2_req,
            )
            # self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
            # self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)

            self.writer = TensorboardWriter(str(self.config.log_dir))

    @staticmethod
    def init_weights(net, init_type="xavier", init_gain=1.4142):
        """Initialize 'net' network weights, based on the chosen 'init_type' and 'init_gain'.

        :param nn.Module net: Network to be initialized.
        :param str init_type: Name of initialization method: normal | xavier | kaiming | orthogonal.
        :param float init_gain: Scaling factor for normal.
        """
        for name, param in net.named_parameters():
            if "weight" in name and "norm" not in name:
                if init_type == "normal":
                    nn.init.normal_(param, mean=0.0, std=init_gain)
                elif init_type == "xavier":
                    nn.init.xavier_uniform_(
                        param, gain=np.sqrt(2.0)
                    )  # ReLU activation function
                elif init_type == "kaiming":
                    nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(
                        param, gain=np.sqrt(2.0)
                    )  # ReLU activation function
                else:
                    raise NotImplementedError(
                        f"initialization method {init_type} is not implemented."
                    )
            elif "bias" in name:
                nn.init.constant_(param, 0.1)

    def length_regularization_loss(self, scores):
        """Compute the summary-length regularization loss based on eq. (1).

        :param torch.Tensor scores: Frame-level importance scores, produced by our CA-SUM model.
        :return: A (torch.Tensor) value indicating the summary-length regularization loss.
        """
        return torch.abs(torch.mean(scores) - self.config.reg_factor)

    def train(self):
        """Main function to train the CA-SUM model."""
        if self.config.verbose:
            tqdm.write("Time to train the model...")

        max_precision, max_recall, max_fscore = 0, 0, 0
        max_epoch = -1

        # min_loss = 1000000000
        # min_epoch = -1
        # min_fscore = 0

        for epoch_i in trange(self.config.n_epochs, desc="Epoch", ncols=80):
            # print(f"epoch: {epoch_i}")
            self.model.train()

            loss_history = []
            # print(f"num train: {len(self.train_loader)}")
            num_batches = int(
                len(self.train_loader) / self.config.batch_size
            )  # full-batch or mini batch
            # print(f"num batch: {num_batches}")
            iterator = iter(self.train_loader)
            for batch_i in trange(num_batches, desc="Batch", ncols=80, leave=False):
                self.optimizer.zero_grad()
                # print(f"batch {batch_i}...")

                for video_i in trange(
                    self.config.batch_size, desc="Video", ncols=80, leave=False
                ):
                    # print(f"video {video_i}...")
                    _, frame_features, change_point = next(iterator)
                    # print(frame_features.shape)
                    frame_features = frame_features.squeeze(0).to(self.config.device)
                    # print(frame_features.shape)

                    output = self.model(frame_features, change_point)
                    # print(f"output: {output}")
                    loss = self.length_regularization_loss(output)
                    loss_history.append(loss.data)
                    loss.backward()

                # Update model parameters every 'batch_size' iterations
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.clip
                )
                self.optimizer.step()
                # self.scheduler.step()

            # Mean loss of each training step
            loss = torch.stack(loss_history).mean()
            if self.config.verbose:
                tqdm.write(f"[{epoch_i}] loss: {loss.item()}")

            # Plot
            if self.config.verbose:
                tqdm.write("Plotting...")
            self.writer.update_loss(loss, epoch_i, "loss_train")

            # Compute fscore on testset
            precision, recall, fscore = self.evaluate(epoch_i, mode="test")
            # Results on test dataset
            self.writer.update_scalar(precision, epoch_i, "precision_test")
            self.writer.update_scalar(recall, epoch_i, "recall_test")
            self.writer.update_scalar(fscore, epoch_i, "fscore_test")

            # # Compute fscore on trainset
            # precision_train, recall_train, fscore_train = self.evaluate(
            #     epoch_i, mode="train"
            # )
            # # Results on train dataset
            # self.writer.update_scalar(precision_train, epoch_i, "precision_train")
            # self.writer.update_scalar(recall_train, epoch_i, "recall_train")
            # self.writer.update_scalar(fscore_train, epoch_i, "fscore_train")

            # if loss < min_loss:
            #     min_loss = loss
            #     min_epoch = epoch_i
            #     min_fscore = fscore
            #     print(
            #         f"Current maximum fscore by loss: epoch {min_epoch}({min_fscore:.2f}%)"
            #     )

            if fscore > max_fscore:
                max_precision = precision
                max_recall = recall
                max_fscore = fscore
                max_epoch = epoch_i
                print(
                    f"Current maximum fscore by test fscore: epoch {max_epoch}({max_fscore:.2f}%)"
                )

                ckpt_path = os.path.join(self.config.model_dir, f"epoch-{epoch_i}.pt")
                tqdm.write(f"Save parameters at {ckpt_path}")
                torch.save(self.model.state_dict(), ckpt_path)

        # return max_epoch, max_fscore, min_epoch, min_loss, min_fscore
        return (
            max_epoch,
            max_precision,
            max_recall,
            max_fscore,
        )

    def evaluate(self, epoch_i, mode="test"):
        """Saves the frame's importance scores for the test videos in json format.

        :param int epoch_i: The current training epoch.
        """
        if self.config.dataset_name == "SumMe":
            inference_method = "max"
        elif self.config.dataset_name == "TVSum":
            inference_method = "avg"

        if mode == "train":
            keys = self.splits[self.config.split_index][mode + "_keys"]
        elif mode == "test":
            keys = self.splits[self.config.split_index][mode + "_keys"]

        precision, recall, fscore = inference(
            self.model,
            keys,
            inference_method,
            self.test_loader.raw_video_features,
            self.test_loader.hdf,
            self.test_loader.dataset_name,
            corr_coef=False,
        )

        # print(f"F1 score on {mode} data: {fscore}")

        return precision, recall, fscore


if __name__ == "__main__":
    pass
