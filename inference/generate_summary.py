# -*- coding: utf-8 -*-
import sys

import numpy as np

np.set_printoptions(threshold=sys.maxsize)
sys.path.append("..")
sys.path.append(".")

from inference.knapsack_implementation import knapSack


def generate_summary(all_shot_bound, all_scores, all_nframes):
    """Generate the automatic machine summary, based on the video shots; the frame importance scores; the number of
    frames in the original video and the position of the sub-sampled frames of the original video.

    :param list[np.ndarray] all_shot_bound: The video shots for all the -original- testing videos.
    :param list[np.ndarray] all_scores: The calculated frame importance scores for all the sub-sampled testing videos.
    :param list[np.ndarray] all_nframes: The number of frames for all the -original- testing videos.
    :param list[np.ndarray] all_positions: The position of the sub-sampled frames for all the -original- testing videos.
    :return: A list containing the indices of the selected frames for all the -original- testing videos.
    """
    all_summaries = []
    # print(len(all_scores[0]))
    # print(all_scores)
    for video_index in range(len(all_scores)):
        # Get shots' boundaries
        shot_bound = all_shot_bound[video_index]  # [number_of_shots, 2]
        shot_bound = [[int(bound[0]), int(bound[1])] for bound in shot_bound]
        shot_scores = all_scores[video_index]  # score per segment
        n_frames = all_nframes[video_index]  #
        shot_lengths = []
        # print("info")
        # print(shot_scores)
        # print(len(shot_bound), len(shot_scores))
        # print(n_frames)

        # # Compute the importance scores for the initial frame sequence (not the sub-sampled one)
        # frame_scores = np.zeros(n_frames, dtype=np.float32)
        for i, shot in enumerate(shot_bound):
            # frame_scores[bound[0] : bound[1]] = shot_init_scores[i]
            length = shot[1] - shot[0] + 1
            shot_lengths.append(length)

        # print(frame_scores)
        # print(shot_bound[0])

        # # Compute shot-level importance scores by taking the average importance scores of all frames in the shot
        # shot_imp_scores = []
        # shot_lengths = []
        # for i, shot in enumerate(shot_bound):
        #     length = shot[1] - shot[0] + 1
        #     shot_lengths.append(length)
        #     shot_imp_scores.append(shot_init_scores[i])
        #
        # Select the best shots using the knapsack implementation
        final_shot = shot_bound[-1]
        # print(final_shot)
        final_max_length = int((final_shot[1] + 1) * 0.15)

        selected = knapSack(
            final_max_length, shot_lengths, shot_scores, len(shot_lengths)
        )

        # Select all frames from each selected shot (by setting their value in the summary vector to 1)
        summary = np.zeros(final_shot[1] + 1, dtype=np.int8)
        summary = np.zeros(n_frames, dtype=np.int8)
        # print(selected)
        for shot in selected:
            summary[shot_bound[shot][0] : shot_bound[shot][1] + 1] = 1
            # summary[shot_bound[shot][0] : shot_bound[shot][1]] = 1

        all_summaries.append(summary)

    return all_summaries
