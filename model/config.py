# -*- coding: utf-8 -*-
import argparse
import os
import pprint

import torch


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


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr."""
        self.log_dir, self.score_dir, self.save_dir = None, None, None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for k, v in kwargs.items():
            setattr(self, k, v)
        # Define paths
        self.split_path = os.path.join(
            kwargs["root_data_path"],
            "splits",
            f"{kwargs['dataset_name'].lower()}_splits.json",
        )
        self.data_path = os.path.join(
            kwargs["root_data_path"],
            kwargs["dataset_name"],
            f"eccv16_dataset_{kwargs['dataset_name'].lower()}_google_pool5.h5",
        )
        self.features_path = os.path.join(
            kwargs["root_data_path"],
            kwargs["dataset_name"],
            f"{kwargs['dataset_name'].lower()}_video_features_version_2.pickle",
        )

    def set_training_dir(self, seed=0, reg_factor=0.6, dataset_name="SumMe"):
        """Function that sets as class attributes the necessary directories for logging important training information.

        :param float reg_factor: The utilized length regularization factor.
        :param str dataset_name: The Dataset being used, SumMe or TVSum.
        """
        self.log_dir = os.path.join(
            self.root_results_path,
            self.experiment_id,
            "reg" + str(reg_factor),
            dataset_name,
            str(seed),
            "logs/split" + str(self.split_index),
        )
        self.score_dir = os.path.join(
            self.root_results_path,
            self.experiment_id,
            "reg" + str(reg_factor),
            dataset_name,
            str(seed),
            "results/split" + str(self.split_index),
        )
        self.model_dir = os.path.join(
            self.root_results_path,
            self.experiment_id,
            "reg" + str(reg_factor),
            dataset_name,
            str(seed),
            "models/split" + str(self.split_index),
        )

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.score_dir):
            os.makedirs(self.score_dir)
        print(self.model_dir)
        print(self.log_dir)
        print(self.score_dir)

    def __repr__(self):
        """Pretty-print configurations in alphabetical order."""
        config_str = "Configurations\n"
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    """Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initialized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        help="Mode for the configuration [train | test]",
    )
    parser.add_argument(
        "--root_data_path",
        type=str,
        default="./data",
        help="Path to training data",
    )
    parser.add_argument(
        "--root_results_path",
        type=str,
        default="./Training_Results/",
        help="Path to training resutls",
    )
    parser.add_argument(
        "--experiment_id",
        type=str,
        default="1",
        help="Experiment id",
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default="false",
        help="Print or not training messages",
    )
    parser.add_argument(
        "--dataset_name", type=str, default="SumMe", help="Dataset to be used"
    )

    # Model
    parser.add_argument(
        "--attn_mechanism",
        type=str2bool,
        default=1,
        help="Whether or not to use the attention mechanism (true or false). Default: true",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=2048,
        help="Feature size expected in the input",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=1,
        help="Size of blocks used inside the attention matrix",
    )
    parser.add_argument(
        "--init_type", type=str, default="xavier", help="Weight initialization method"
    )
    parser.add_argument(
        "--init_gain",
        type=float,
        default=1.4142,
        help="Scaling factor for the initialization methods",
    )
    parser.add_argument(
        "--seg_emb_method",
        type=str,
        default="mean",
        help="Method to extract information from segment (max, mean, attention). Default: mean",
    )

    # Train
    parser.add_argument(
        "--n_epochs", type=int, default=200, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="Size of each batch in training. Default: 20 videos/batch",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Chosen seed for generating random numbers",
    )
    parser.add_argument(
        "--clip", type=float, default=5.0, help="Max norm of the gradients"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="Learning rate used for the modules"
    )
    parser.add_argument(
        "--l2_req", type=float, default=1e-5, help="Weight regularization factor"
    )
    parser.add_argument(
        "--reg_factor", type=float, default=0.6, help="Length regularization factor"
    )
    parser.add_argument(
        "--split_index", type=int, default=0, help="Data split to be used [0-4]"
    )

    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)


if __name__ == "__main__":
    config = get_config()
    print(config)
    # import ipdb
    # ipdb.set_trace()
