# -*- coding: utf-8 -*-
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

debug = False


class SegAttention(nn.Module):
    def __init__(self, input_size=1024, output_size=1024):
        super(SegAttention, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.Wk = nn.Linear(
            in_features=input_size, out_features=output_size, bias=False
        )  # key using mean
        self.Wq = nn.Linear(
            in_features=input_size, out_features=output_size, bias=False
        )
        self.Wv = nn.Linear(
            in_features=input_size, out_features=output_size, bias=False
        )
        self.out = nn.Linear(
            in_features=input_size, out_features=output_size, bias=False
        )

        self.layernorm = nn.LayerNorm(normalized_shape=input_size, eps=1e-6)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_diff = x[0 : x.shape[0] - 1, :] - x[1 : x.shape[0], :]

        seg_mean = torch.mean(x_diff, dim=0).reshape(1, -1)
        K = self.Wk(seg_mean)
        Q = self.Wq(x_diff)
        V = self.Wv(x_diff)
        # print(K.shape, Q.shape, V.shape)

        energies = torch.matmul(K, Q.transpose(1, 0))
        # print(f"energies: {energies.shape}")
        scores = self.softmax(energies)
        # print(f"scores: {scores.shape}")
        seg_embedding = torch.matmul(scores, V)
        # print(f"seg_embedding: {seg_embedding.shape}")

        seg_embedding += seg_mean
        norm_seg_embedding = self.layernorm(seg_embedding)

        return norm_seg_embedding


class SelfAttention(nn.Module):
    def __init__(self, input_size=1024, output_size=1024, block_size=60):
        """The basic Attention 'cell' containing the learnable parameters of Q, K and V.

        :param int input_size: Feature input size of Q, K, V.
        :param int output_size: Feature -hidden- size of Q, K, V.
        :param int block_size: The size of the blocks utilized inside the attention matrix.
        """
        super(SelfAttention, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.block_size = block_size
        self.Wk = nn.Linear(
            in_features=input_size, out_features=output_size, bias=False
        )
        self.Wq = nn.Linear(
            in_features=input_size, out_features=output_size, bias=False
        )
        self.Wv = nn.Linear(
            in_features=input_size, out_features=output_size, bias=False
        )
        self.out = nn.Linear(
            in_features=output_size + 2, out_features=input_size, bias=False
        )

        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def get_entropy(logits):
        """Compute the entropy for each row of the attention matrix.

        :param torch.Tensor logits: The raw (non-normalized) attention values with shape [T, T].
        :return: A torch.Tensor containing the normalized entropy of each row of the attention matrix, with shape [T].
        """
        _entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
        _entropy = -1.0 * _entropy.sum(-1)

        # https://stats.stackexchange.com/a/207093 Maximum value of entropy is log(k), where k the # of used categories.
        # Here k is when all the values of a row is different of each other (i.e., k = # of video frames)
        return _entropy / np.log(logits.shape[0])

    def forward(self, x):
        """Compute the weighted frame features, through the Block diagonal sparse attention matrix and the estimates of
        the frames attentive uniqueness and the diversity.

        :param torch.Tensor x: Frame features with shape [T, input_size].
        :return: A tuple of:
                    y: The computed weighted features, with shape [T, input_size].
                    att_win : The Block diagonal sparse attention matrix, with shape [T, T].
        """
        # Compute the pairwise dissimilarity of each frame, on the initial feature space (GoogleNet features)
        if debug:
            print(x)
        x_unit = F.normalize(x, p=2, dim=1)
        if debug:
            print(x_unit)
        similarity = x_unit @ x_unit.t()  # dot product
        if debug:
            print("sim:", similarity.shape)
        diversity = 1 - similarity  # raw diversity matrix
        if debug:
            print("diver:", diversity.shape)

        K = self.Wk(x)
        Q = self.Wq(x)
        V = self.Wv(x)
        if debug:
            print("kqv:")
            print(K.shape)
            print(Q.shape)
            print(V.shape)

        energies = torch.matmul(Q, K.transpose(1, 0))
        att_weights = self.softmax(energies)  # attention matrix
        if debug:
            print(energies.shape)
            print(att_weights.shape)
            print(att_weights)

        # Entropy is a measure of uncertainty: Higher value means less information.
        entropy = self.get_entropy(logits=energies)
        entropy = F.normalize(entropy, p=1, dim=-1)  # unique vector
        if debug:
            print(entropy.shape)

        # Compute the mask to form the Block diagonal sparse attention matrix
        D = self.block_size
        num_blocks = math.ceil(energies.shape[0] / D)  # 35/4=9 blocks
        keepingMask = torch.ones(num_blocks, D, D, device=att_weights.device)
        if debug:
            print(keepingMask)
            print(keepingMask.shape)
        keepingMask = torch.block_diag(*keepingMask)[
            : att_weights.shape[0], : att_weights.shape[0]
        ]
        if debug:
            print(keepingMask)
            print(keepingMask.shape)
        zeroingMask = 1 - keepingMask
        if debug:
            print(zeroingMask)
            print(zeroingMask.shape)
        att_win = att_weights * keepingMask  # attention inside block
        if debug:
            print(att_win)
            print(att_win.shape)

        # Pick those frames that are "invisible" to a frame, aka outside the block (mask)
        attn_remainder = att_weights * zeroingMask  # attention outside block
        div_remainder = diversity * zeroingMask  # diversity outside block

        if debug:
            print(attn_remainder)
            print(attn_remainder.shape)
            print(div_remainder)
            print(div_remainder.shape)

        # Compute non-local dependencies based on the diversity of those frames
        dep_factor = (
            (div_remainder * attn_remainder).sum(-1).div(div_remainder.sum(-1))
        )  # dependencies of outside block, also diversity vector
        if debug:
            print("dep_factor")
            print(dep_factor.shape)
            print(dep_factor)
        dep_factor = dep_factor.unsqueeze(0).expand(dep_factor.shape[0], -1)
        if debug:
            print(dep_factor.shape)
            print(dep_factor)
        masked_dep_factor = dep_factor * keepingMask
        if debug:
            print("mask")
            print(masked_dep_factor)
        att_win += masked_dep_factor  # final block diagonal sparse attetion matrix

        y = torch.matmul(att_win, V)
        if debug:
            print(y)
            print(y.shape)
        characteristics = (entropy, dep_factor[0, :])  # stack unique + diveristy
        characteristics = torch.stack(characteristics).detach()
        if debug:
            print(characteristics)
        outputs = torch.cat(tensors=(y, characteristics.t()), dim=-1)
        if debug:
            print(outputs)
            print(outputs.shape)

        y = self.out(outputs)
        if debug:
            print(y)
            print(y.shape)
        return y, att_win.clone()


if __name__ == "__main__":
    pass
    """Uncomment for a quick proof of concept"""
    model = SelfAttention(input_size=8, output_size=4, block_size=2).cuda()
    _input = torch.randn(5, 8).cuda()  # [seq_len, hidden_size]
    output, weights = model(_input)
    print(f"Output shape: {output.shape}\tattention shape: {weights.shape}")
