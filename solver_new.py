# -*- coding: utf-8 -*-
import json
import os
import random

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from inference.inference_new import inference
# from model.summarizer import Sum
from model.new_sum import Sum
from tqdm import tqdm, trange
from utils import TensorboardWriter


class Solver(object):
    def __init__(self, config=None, train_loader=None, test_loader=None):
        """Class that Builds, Trains and Evaluates CA-SUM model."""
        # Initialize variables to None, to be safe
        self.model, self.optimizer, self.writer = None, None, None

        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

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

        max_fscore = 0
        max_epoch = -1

        min_loss = 1000000000
        min_epoch = -1
        min_fscore = 0

        for epoch_i in trange(self.config.n_epochs, desc="Epoch", ncols=80):
            self.model.train()

            loss_history = []
            num_batches = int(
                len(self.train_loader) / self.config.batch_size
            )  # full-batch or mini batch
            iterator = iter(self.train_loader)
            for _ in trange(num_batches, desc="Batch", ncols=80, leave=False):
                self.optimizer.zero_grad()

                for _ in trange(
                    self.config.batch_size, desc="Video", ncols=80, leave=False
                ):
                    _, frame_features, change_point = next(iterator)
                    frame_features = frame_features.squeeze(0).to(self.config.device)

                    # output, _ = self.model(frame_features)
                    output = self.model(frame_features, change_point)
                    loss = self.length_regularization_loss(output)
                    loss_history.append(loss.data)
                    loss.backward()

                # Update model parameters every 'batch_size' iterations
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.clip
                )
                self.optimizer.step()

            # Mean loss of each training step
            loss = torch.stack(loss_history).mean()
            if self.config.verbose:
                tqdm.write(f"[{epoch_i}] loss: {loss.item()}")

            # Plot
            if self.config.verbose:
                tqdm.write("Plotting...")
            self.writer.update_loss(loss, epoch_i, "loss_epoch")

            # Compute fscore on testset
            fscore = self.evaluate(epoch_i)

            if loss < min_loss:
                min_loss = loss
                min_epoch = epoch_i
                min_fscore = fscore
                print(
                    f"Current maximum fscore by loss: epoch {max_epoch}({max_fscore:.2f}%)"
                )

            if fscore > max_fscore:
                max_fscore = fscore
                max_epoch = epoch_i
                print(
                    f"Current maximum fscore by test fscore: epoch {max_epoch}({max_fscore:.2f}%)"
                )

                # if not os.path.exists(self.config.save_dir):
                #     os.makedirs(self.config.save_dir)
                parent_path = f"./trained_model/{self.config.split_index}"
                if not os.path.exists(parent_path):
                    os.makedirs(parent_path)
                ckpt_path = parent_path + f"/epoch-{epoch_i}.pt"
                tqdm.write(f"Save parameters at {ckpt_path}")
                torch.save(self.model.state_dict(), ckpt_path)

        return max_epoch, max_fscore, min_epoch, min_loss, min_fscore

    def evaluate(self, epoch_i):
        """Saves the frame's importance scores for the test videos in json format.

        :param int epoch_i: The current training epoch.
        """

        fscore = inference(
            self.model,
            self.test_loader.keys,
            "max",
            self.test_loader.raw_video_features,
            self.test_loader.hdf,
            "SumMe",
            False,
        )
        return fscore


if __name__ == "__main__":
    pass
