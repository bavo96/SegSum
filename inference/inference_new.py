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
from inference.generate_summary_new import generate_summary
# from model.summarizer import CA_SUM
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


def inference(model, keys, eval_method, raw_video_features, hdf, dataset, corr_coef):
    """Used to inference a pretrained `model` on the `keys` test videos, based on the `eval_method` criterion; using
    the dataset located in `data_path'.

    :param nn.Module model: Pretrained model to be inferenced.
    :param str data_path: File path for the dataset in use.
    :param list keys: Containing the test video keys of the used data split.
    :param str eval_method: The evaluation method in use {SumMe: max, TVSum: avg}.
    """
    model.eval()
    video_fscores, video_rho, video_tau = [], [], []

    for video in keys:
        video_fullname = np.array(hdf.get(video + "/video_name")).astype("str").tolist()
        frame_features = torch.squeeze(torch.stack(raw_video_features[video_fullname]))

        user_summary = np.array(hdf[f"{video}/user_summary"])
        n_frames = np.array(hdf[f"{video}/n_frames"])

        sb = torch.Tensor(np.array(hdf[f"{video}/change_points"]))

        # # Custom sb
        # def divide_chunks(l, n):
        #     # looping till length l
        #     for i in range(0, len(l), n):
        #         yield l[i : i + n]
        # num_frame = frame_features.shape[0]
        # l_frame = [i for i in range(num_frame)]
        # l_frame = list(divide_chunks(l_frame, 30))
        # change_point = [[item[0], item[-1]] for item in l_frame]
        # sb = torch.Tensor(change_point)

        with torch.no_grad():
            # scores, _ = model(frame_features)  # [1, seq_len]
            scores = model(frame_features, sb)  # [1, seq_len]
            scores = scores.squeeze(0).cpu().numpy().tolist()

            summary = generate_summary([sb], [scores], [n_frames])[0]
            f_score = evaluate_summary(summary, user_summary, eval_method)
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

    return np.mean(video_fscores)


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
    dataset = "SumMe"  # args["dataset"]
    corr_coef = False  # args["corr_coef"]

    eval_metric = "avg" if dataset.lower() == "tvsum" else "max"
    l_fscore = []

    # Get raw features
    raw_data_path = "./data/video_features.pickle"
    start = time.time()
    raw_video_features = pickle.load(
        open(raw_data_path, "rb"),
    )
    end = time.time() - start
    print("Load raw features:", end)

    # Dataset path
    dataset_path = f"./data/{dataset}/eccv16_dataset_{dataset.lower()}_google_pool5.h5"
    hdf = h5py.File(dataset_path, "r")  # Open hdf file

    for split_id in range(5):
        # Model data
        # model_path = f"./inference/pretrained_models/{dataset}/split{split_id}"
        # model_path = "./Summaries/exp1/reg0.6/SumMe/"
        model_path = f"./trained_model/{split_id}/"
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
        trained_model = Sum(input_size=2048, output_size=2048, block_size=2).to(device)
        # trained_model = CA_SUM(input_size=1024, output_size=1024, block_size=60).to(
        #     device
        # )
        trained_model.load_state_dict(torch.load(model_full_path))
        f_score = inference(
            trained_model,
            test_keys,
            eval_metric,
            raw_video_features,
            hdf,
            dataset,
            corr_coef,
        )
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
    print("Mean:", np.mean(l_fscore))
