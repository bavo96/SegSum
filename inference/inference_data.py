# -*- coding: utf-8 -*-
import sys

sys.path.append("..")
sys.path.append(".")

import argparse
import json
import pickle
import time
from os import listdir
from os.path import isfile, join

import h5py
import numpy as np
import torch

from inference.evaluation_metrics import evaluate_summary, get_corr_coeff
from inference.generate_summary import generate_summary
from model.new_sum import Sum

eligible_datasets = ["TVSum"]


def str2bool(v):
    """Transcode string to boolean.

    :param str v: String to be transcoded.
    :return: The boolean transcoding of the string.
    """
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i : i + n]


def inference(model, keys, eval_method, raw_video_features, hdf, dataset, corr_coef):
    """Used to inference a pretrained `model` on the `keys` test videos, based on the `eval_method` criterion; using
    the dataset located in `data_path'.

    :param nn.Module model: Pretrained model to be inferenced.
    :param str data_path: File path for the dataset in use.
    :param list keys: Containing the test video keys of the used data split.
    :param str eval_method: The evaluation method in use {SumMe: max, TVSum: avg}.
    """
    model.eval()
    video_precisions, video_recalls, video_fscores, video_rho, video_tau = (
        [],
        [],
        [],
        [],
        [],
    )

    video_segments_scores = {}

    for video in keys:
        ## Get videos' names
        if dataset == "SumMe":
            video_fullname = (
                np.array(hdf.get(video + "/video_name")).astype("str").tolist()
            )
        elif dataset == "TVSum":
            frame2video = {}
            video2index = {}
            index2video = {}
            for key in list(hdf.keys()):
                nframes = int(np.array(hdf[key]["n_frames"]))
                full_name = np.array(hdf[key]["n_frames"]).astype("str").tolist()
                frame2video[nframes] = key

            for k, v in raw_video_features.items():
                video_fullname = k
                video2index[video_fullname] = frame2video[v.shape[0]]
                index2video[frame2video[v.shape[0]]] = video_fullname
            video_fullname = index2video[video]

        frame_features = raw_video_features[video_fullname]

        user_summary = np.array(hdf[f"{video}/user_summary"])
        n_frames = np.array(hdf[f"{video}/n_frames"])
        sb = torch.Tensor(np.array(hdf[f"{video}/change_points"]))

        # # Custom sb
        # l_frame = [i for i in range(n_frames)]
        # l_frame = list(divide_chunks(l_frame, 60))
        # change_point = [[item[0], item[-1]] for item in l_frame]
        # sb = torch.Tensor(change_point)

        # print("inference...")
        with torch.no_grad():
            scores = model(frame_features, sb)  # [1, seq_len]
            # print(sb.shape, scores.shape, n_frames, user_summary.shape, eval_method)
            # print(f"Segment shape: {scores.shape}")
            scores = scores.squeeze(0).cpu().numpy().tolist()
            # print(scores)

            summary = generate_summary([sb], [scores], [n_frames])[0]
            video_segments_scores[video_fullname] = summary
            if dataset == "SumMe":
                (precision, recall, f_score), best_user = evaluate_summary(
                    summary, user_summary, eval_method
                )
                video_segments_scores[f"best_user_{video_fullname}"] = best_user
                # print(f"Score for {video_fullname}: ", precision, recall, f_score)
            elif dataset == "TVSum":
                precision, recall, f_score = evaluate_summary(
                    summary, user_summary, eval_method
                )

            video_precisions.append(precision)
            video_recalls.append(recall)
            video_fscores.append(f_score)

            # print("length vid:", len(summary))
            # print("sum length:", np.count_nonzero(summary == 1))
            # print("percentage:", np.count_nonzero(summary == 1) / len(summary))

            if dataset in eligible_datasets and corr_coef:
                rho, tau = get_corr_coeff(
                    pred_imp_scores=scores, video=video, dataset=dataset
                )
                video_rho.append(rho)
                video_tau.append(tau)

    # Save scores to pickle
    with open(f"./analyze/{dataset.lower()}_video_scores.pickle", "wb") as handle:
        pickle.dump(video_segments_scores, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return np.mean(video_precisions), np.mean(video_recalls), np.mean(video_fscores)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # arguments to run the script
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="SumMe",
        help="Dataset to be used. Supported: {SumMe, TVSum}",
    )
    parser.add_argument(
        "--corr_coef",
        type=str2bool,
        default=False,
        help="Calculate or not, the correlation coefficients",
    )

    args = vars(parser.parse_args())
    # dataset = "TVSum"  # args["dataset"]
    dataset = "SumMe"  # args["dataset"]
    corr_coef = False  # args["corr_coef"]
    seed = 3
    eval_metric = "avg" if dataset.lower() == "tvsum" else "max"
    print(f"Dataset: {dataset}, eval_metric: {eval_metric}")

    l_precision, l_recall, l_fscore = [], [], []

    # Get raw features
    raw_video_features_path = (
        f"./data/{dataset}/{dataset.lower()}_video_features.pickle"
    )
    print(raw_video_features_path)
    start = time.time()
    raw_video_features = pickle.load(
        open(raw_video_features_path, "rb"),
    )
    end = time.time() - start
    print("Load raw features:", end)

    # Dataset path
    dataset_path = f"./data/{dataset}/eccv16_dataset_{dataset.lower()}_google_pool5.h5"
    hdf = h5py.File(dataset_path, "r")  # Open hdf file

    for split_id in range(5):
        # Model data
        # model_path = f"./Summaries/reg0.15/{dataset}/{seed}/models/split{split_id}/"
        model_path = f"./Summaries/reg0.15_1/{dataset}/{seed}/models/split{split_id}/"
        # model_path = f"./results/{dataset.lower()}_trained_model/best_model/{split_id}/"
        # model_path = f"./results/{dataset.lower()}_trained_model/{split_id}/"
        print(model_path)
        model_file = max(
            [
                int(f.split(".")[0].split("-")[1])
                for f in listdir(model_path)
                if isfile(join(model_path, f))
            ]
        )
        model_full_path = join(model_path, f"epoch-{model_file}.pt")
        # print("model's path:", model_full_path)

        # Read current split
        split_file = f"./data/splits/{dataset.lower()}_splits.json"
        with open(split_file) as f:
            data = json.loads(f.read())
            test_keys = data[split_id]["test_keys"]

        # Create model with paper reported configuration
        trained_model = Sum(input_size=2048, output_size=2048, block_size=1).to(device)
        # trained_model = CA_SUM(input_size=1024, output_size=1024, block_size=60).to(
        #     device
        # )
        trained_model.load_state_dict(torch.load(model_full_path))
        precision, recall, f_score = inference(
            trained_model,
            test_keys,
            eval_metric,
            raw_video_features,
            hdf,
            dataset,
            corr_coef,
        )
        l_precision.append(precision)
        l_recall.append(recall)
        l_fscore.append(f_score)

        print(
            f"CA-SUM model trained for split: {split_id} achieved an F-score: {f_score:.2f}%"
        )
        # if dataset not in eligible_datasets or not corr_coef:
        #     print("\n", end="")
        # else:
        #     print(
        #         f", a Spearman's \u03C1: {np.mean(video_rho):.3f}  and a Kendall's \u03C4: {np.mean(video_tau):.3f}"
        #     )

    hdf.close()

    print("Standard deviation:", np.std(l_fscore))
    print("Variance:", np.var(l_fscore))
    print("Mean Precision:", np.mean(l_precision))
    print("Mean Recall:", np.mean(l_recall))
    print("Mean F-Score:", np.mean(l_fscore))
