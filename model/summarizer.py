# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn

import time
from .attention import SelfAttention, SegAttention

# from attention import SelfAttention

debug = False


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]


class Sum(nn.Module):
    def __init__(
        self,
        input_size=1024,
        output_size=1024,
        block_size=60,
        seg_method="max",
        attn_mechanism=True,
    ):
        """Class wrapping the CA-SUM model; its key modules and parameters.

        :param int input_size: The expected input feature size.
        :param int output_size: The produced output feature size.
        :param int block_size: The size of the blocks utilized inside the attention matrix.
        """
        super(Sum, self).__init__()

        self.seg_method = seg_method
        self.attn_mechanism = attn_mechanism

        if self.attn_mechanism:
            self.attention = SelfAttention(
                input_size=input_size, output_size=output_size, block_size=block_size
            )

        if self.seg_method == "attention":
            self.segattention = SegAttention(
                input_size=input_size, output_size=output_size
            )
        self.linear_1 = nn.Linear(in_features=input_size, out_features=input_size)
        self.linear_2 = nn.Linear(
            in_features=self.linear_1.out_features, out_features=1
        )

        self.drop = nn.Dropout(p=0.5)
        self.norm_y = nn.LayerNorm(normalized_shape=input_size, eps=1e-6)
        self.norm_linear = nn.LayerNorm(
            normalized_shape=self.linear_1.out_features, eps=1e-6
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, frame_features, change_point):
        """Produce frame-level importance scores from the frame features, using the CA-SUM model.

        :param torch.Tensor frame_features: Tensor of shape [T, input_size] containing the frame features produced by
        using the pool5 layer of GoogleNet.
        :return: A tuple of:
            y: Tensor with shape [1, T] containing the frames importance scores in [0, 1].
            attn_weights: Tensor with shape [T, T] containing the attention weights.
        """
        list_segments = []

        ## CHANGE POINT
        change_point = torch.squeeze(change_point)

        # -- Generate custom change point --
        # num_frame = frame_features.shape[0]
        # l_frame = [i for i in range(num_frame)]
        # l_frame = list(divide_chunks(l_frame, 60))
        # change_point = [[item[0], item[-1]] for item in l_frame]
        # change_point = torch.Tensor(change_point)
        # -- Get original change point
        for cp in change_point:
            cp = cp.numpy()
            seg_feat = frame_features[int(cp[0]) : int(cp[1]) + 1, :]
            list_segments.append(seg_feat)
        if debug:
            print("segment shape:")
            print(len(list_segments))
            for seg in list_segments:
                print(seg.shape)

        ## SEGMENT EMBEDDING
        # start = time.time()
        if self.seg_method == "max":
            new_list = []
            for seg in list_segments:
                seg_emb = torch.max(seg, dim=0).values
                seg_emb = seg_emb.reshape(1, -1)  # max tensor, max indices
                new_list.append(seg_emb)
            segments = torch.cat(new_list, 0)
        # -- mean --
        elif self.seg_method == "mean":
            new_list = []
            for seg in list_segments:
                seg_emb = torch.mean(seg, dim=0)
                seg_emb = seg_emb.reshape(1, -1)  # mean tensor
                new_list.append(seg_emb)
            segments = torch.cat(new_list, 0)
        # -- attention --
        elif self.seg_method == "attention":
            new_list = []
            for seg in list_segments:
                seg_emb = self.segattention(seg)
                new_list.append(seg_emb)
            segments = torch.cat(new_list, 0)
        # end = time.time() - start
        # print(f"seg emb time: {end}")
        if debug:
            print("final shape:")
            print(segments.shape)

        ## SELF ATTENTION
        if self.attn_mechanism:
            weighted_value, attn_weights = self.attention(
                segments
            )  # weighted_value shape: N segments x M features
            y = segments + weighted_value
        else:
            y = segments

        ## REGRESSOR NETWORK
        y = self.drop(y)
        y = self.norm_y(y)

        # 2-layer NN (Regressor Network)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.drop(y)
        y = self.norm_linear(y)

        y = self.linear_2(y)
        if debug:
            print("yshape:", y.shape)
        y = self.sigmoid(y)
        y = y.view(1, -1)
        if debug:
            print("yshape:", y.shape)

        # return y, attn_weights
        return y


if __name__ == "__main__":
    import random

    random.seed(1)

    _input = torch.randn(500, 256).cuda()  # [seq_len, hidden_size]
    random_cp = sorted(random.sample(range(0, 500), 5))
    change_point = []
    change_point.append((0, random_cp[0]))
    for i in range(len(random_cp) - 1):
        change_point.append((random_cp[i], random_cp[i + 1]))

    change_point = torch.Tensor(np.array(change_point))

    model = Sum(input_size=256, output_size=256, block_size=2).cuda()
    output = model(_input, change_point)

    print(f"output: {output}")
