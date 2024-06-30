import json
import os
import pickle
import time
from collections import defaultdict

import cv2
import gradio as gr
import h5py
import numpy as np
import torch
from tqdm import tqdm

from inference.generate_summary import generate_summary
from model.summarizer import Sum

# SumMe: max method, sigma=0.3, seed=6, blocksize = 1, split 2
# data/SumMe/videos/

# TVSum: avg method, sigma=0.9, seed=2, blocksize = 2, split 4
# data/tvsum50_ver_1_1/ydata-tvsum50-v1_1/ydata-tvsum50-video/video/


def visualize(dataset, video_key, video_fullname, fps, model_score):
    # CA-SUM data
    user_summary = np.array(dataset_dict[dataset]["hdf"][video_key]["user_summary"])
    n_frames = np.array(dataset_dict[dataset]["hdf"][video_key]["n_frames"])
    change_points = np.array(dataset_dict[dataset]["hdf"][video_key]["change_points"])

    # kts_segments = []
    # len_kts_seg = []

    s = 1
    flag = 1

    # Get change point of KTS
    print(f"Number of changepoint: {len(change_points)}")
    # for cp in change_points:
    #     num_frame = cp[1] - cp[0] + 1
    #     len_kts_seg.append(num_frame)
    #
    #     for i in range(num_frame):
    #         kts_segments.append(s)
    #
    #     s += flag
    #     flag *= -1

    # Create result image
    img = np.ones((950, n_frames, 4), np.uint8) * 255

    # Draw kts
    cv2.rectangle(img, (0, 750), (int(n_frames), 800), (143, 255, 255, 0), -1)

    for i, cp in enumerate(change_points):
        if i == len(change_points) - 1:
            break
        # num_frame = cp[1] - cp[0] + 1
        cv2.line(img, (cp[1], 750), (cp[1], 800), (0, 128, 139, 0), thickness=5)

    # Draw model scores
    video_score = model_score
    for i in range(n_frames):
        if video_score[i] == 0:
            img[810:860, i] = (255, 0, 0, 0)  # Blue
        elif video_score[i] == 1:
            img[810:860, i] = (0, 255, 0, 0)  # Green
        else:
            print(f"video_score:{video_score[i]}")
            img[810:860, i] = (0, 255, 255, 0)

    # Draw time range
    for i in range(n_frames):
        if i % (fps * 5) == 0:
            img = cv2.putText(
                img,
                str(int(i / fps)),
                (i, 890),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0, 0),
                2,
            )

    # Draw user's segments
    len_user_seg = []
    num_user = user_summary.shape[0]
    for user in range(user_summary.shape[0]):
        color = (0, 0, 255, 0)
        user_sum = [
            i if user_summary[user][i] != 0 else 0
            for i in range(len(user_summary[user]))
        ]

        # Get user segments to do analysis
        user_segment = []
        for i in range(len(user_sum)):
            if (i == 0 and user_sum[i] != 0) or (
                i != 0 and user_sum[i - 1] == 0 and user_sum[i] != 0
            ):
                start = user_sum[i]
            elif (i == len(user_sum) - 1 and user_sum[i] != 0) or (
                i != len(user_sum) - 1 and user_sum[i + 1] == 0 and user_sum[i] != 0
            ):
                end = user_sum[i]
                len_user_seg.append(end - start + 1)
                user_segment.append([start, end])

        # Draw user score
        for i in range(n_frames):
            if user_summary[user][i] == 1:
                img[
                    int(700 / num_user) * (user + 1) : int(700 / num_user) * (user + 2)
                    - 10,
                    i,
                ] = color

    # Draw scores
    res_path = "./web_app/visualize_temp.jpeg"
    cv2.imwrite(res_path, img)


def extractImages(pathIn, pathOut, summary, nframes):
    print(
        f"{pathIn}, {os.path.exists(pathIn)}",
    )
    in_video = cv2.VideoCapture(pathIn)
    inwidth = int(in_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    inheight = int(in_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    infps = int(in_video.get(cv2.CAP_PROP_FPS))
    out_video = cv2.VideoWriter(
        "./web_app/temp.mp4",
        cv2.VideoWriter_fourcc(*"MP4V"),
        infps,
        (inwidth, inheight),
    )

    success, image = in_video.read()
    count = 0
    success = True
    count_sum = 0
    with tqdm(total=nframes) as pbar:
        while success and count < nframes:
            success, image = in_video.read()
            if summary[count] == 1:
                count_sum += 1
                out_video.write(image)
            count += 1
            pbar.update(1)

    print(f"Total frames: {count}, Summary frames: {count_sum}")

    in_video.release()
    out_video.release()

    cv2.destroyAllWindows()

    # Visualize result
    return infps


def inference(dataset, video_name, video_fullname):
    dataset_dict[dataset]["trained_model"].eval()
    video_segments_scores = defaultdict(dict)

    frame_features = dataset_dict[dataset]["raw_video_features"][video_name]

    if dataset == "TVSum":
        key = dataset_dict[dataset]["video2index"][video_name]
    elif dataset == "SumMe":
        key = dataset_dict[dataset]["video2index"][video_name]
    n_frames = np.array(dataset_dict[dataset]["hdf"][f"{key}/n_frames"])
    sb = torch.Tensor(np.array(dataset_dict[dataset]["hdf"][f"{key}/change_points"]))

    with torch.no_grad():
        scores, attn_weights = dataset_dict[dataset]["trained_model"](
            frame_features, sb
        )  # [1, seq_len]
        scores = scores.squeeze(0).cpu().numpy().tolist()
        video_segments_scores[video_fullname]["score"] = scores

        summary = generate_summary([sb], [scores], [n_frames])[0]
        print(n_frames, summary.shape)
        fps = extractImages(video_fullname, "./web_app/temp.mp4", summary, n_frames)

        visualize(dataset, key, video_fullname, fps, summary)


def gr_video_name_select(dataset, video_name):
    if dataset == "SumMe":
        raw_video_fullname = "../../data/SumMe/videos/" + video_name + ".mp4"
    elif dataset == "TVSum":
        raw_video_fullname = (
            "../../data/tvsum50_ver_1_1/ydata-tvsum50-v1_1/ydata-tvsum50-video/video/"
            + video_name
            + ".mp4"
        )
    print(
        f"{video_name}, {raw_video_fullname}, {os.path.exists(raw_video_fullname)}",
    )
    print(
        f"{raw_video_fullname}, {os.path.exists(raw_video_fullname)}",
    )

    return video_name, raw_video_fullname


def gr_generate_summary_click(dataset, video_name, raw_video_fullname):
    print(
        f"{raw_video_fullname}, {os.path.exists(raw_video_fullname)}",
    )

    inference(dataset, video_name, raw_video_fullname)

    return "./web_app/temp.mp4", "./web_app/visualize_temp.jpeg"


def gr_dataset_select(dataset):
    print(dataset_dict[dataset]["list_video_name"])
    list_video_name = dataset_dict[dataset]["list_video_name"]
    return gr.update(choices=list_video_name)


def initialization(datasets):
    dataset_dict = {}

    for dataset in datasets:
        dataset_dict[dataset] = {}
        split_id = 4 if dataset.lower() == "tvsum" else 2
        seed = 2 if dataset.lower() == "tvsum" else 6
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Dataset: {dataset}")
        # Get raw features
        raw_video_features_path = (
            f"./data/{dataset}/{dataset.lower()}_video_features_version_2.pickle"
        )
        start = time.time()
        raw_video_features = pickle.load(
            open(raw_video_features_path, "rb"),
        )
        print(f"Raw features path: {raw_video_features_path}")
        end = time.time() - start
        print("Load raw features:", end)
        dataset_dict[dataset]["raw_video_features"] = raw_video_features

        # Dataset path
        dataset_path = (
            f"./data/{dataset}/eccv16_dataset_{dataset.lower()}_google_pool5.h5"
        )
        hdf = h5py.File(dataset_path, "r")  # Open hdf file
        dataset_dict[dataset]["hdf"] = hdf

        # Model data
        model_path = f"./inference/best_models/{dataset}/{seed}/models/split{split_id}"
        model_file = max(
            [
                int(f.split(".")[0].split("-")[1])
                for f in os.listdir(model_path)
                if os.path.isfile(os.path.join(model_path, f))
            ]
        )
        model_full_path = os.path.join(model_path, f"epoch-{model_file}.pt")
        print("Model's path:", model_full_path)

        # Read current split
        split_file = f"./data/splits/{dataset.lower()}_splits.json"
        with open(split_file) as f:
            data = json.loads(f.read())
            # keys = data[split_id]["train_keys"]
            keys = data[split_id]["test_keys"]

        dataset_dict[dataset]["keys"] = keys

        # Create model with paper reported configuration
        if dataset == "SumMe":
            trained_model = Sum(
                input_size=1024,
                output_size=1024,
                block_size=1,
                attn_mechanism=True,
                seg_method="max",
            ).to(device)
        elif dataset == "TVSum":
            trained_model = Sum(
                input_size=1024,
                output_size=1024,
                block_size=2,
                attn_mechanism=True,
                seg_method="max",
            ).to(device)

        trained_model.load_state_dict(torch.load(model_full_path))
        dataset_dict[dataset]["trained_model"] = trained_model

        list_video_name = []
        if dataset == "TVSum":
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

            for video in keys:
                video_fullname = index2video[video]
                list_video_name.append(video_fullname)

        elif dataset == "SumMe":
            video2index = {}
            index2video = {}
            for video in keys:
                video_fullname = (
                    np.array(hdf.get(video + "/video_name")).astype("str").tolist()
                )
                video2index[video_fullname] = video
                index2video[video] = video_fullname
                list_video_name.append(video_fullname)

        dataset_dict[dataset]["video2index"] = video2index
        dataset_dict[dataset]["index2video"] = index2video
        dataset_dict[dataset]["list_video_name"] = list_video_name

    return dataset_dict


if __name__ == "__main__":
    dataset = ["TVSum", "SumMe"]  # args["dataset"]

    dataset_dict = initialization(dataset)

    with gr.Blocks() as demo:
        gr_dataset = gr.Radio(["SumMe", "TVSum"], value="TVSum", label="Dataset")
        gr_video_name = gr.Radio(
            dataset_dict["TVSum"]["list_video_name"], label="Video name"
        )
        gr_selected_video = gr.Textbox(label="Selected video", interactive=False)
        with gr.Row():
            with gr.Column():
                gr_full_video = gr.PlayableVideo(
                    label="Full video", interactive=False, format="mp4", height=480
                )
                gr_generate_summary = gr.Button(value="Generate summary")
            with gr.Column():
                gr_summary_video = gr.PlayableVideo(
                    label="Summary video", interactive=False, format="mp4", height=480
                )

        gr_visualization = gr.Image(
            show_label=False,
            label="Result's visualiation",
            interactive=False,
            type="filepath",
        )
        gr_description = gr.Textbox(
            label="Description",
            value="The red bars represent human frame selections for the video, with each row indicating a human summary. The pastel yellow bar depicts the KTS segment for the video, marked by a small dark yellow line. The green bar illustrates our model’s frame selection for the summary, while the blue bar  represents frames that are not selected.",
            interactive=False,
            max_lines=100,
        )

        gr_dataset.select(
            fn=gr_dataset_select, inputs=[gr_dataset], outputs=[gr_video_name]
        )
        gr_video_name.select(
            fn=gr_video_name_select,
            inputs=[gr_dataset, gr_video_name],
            outputs=[gr_selected_video, gr_full_video],
        )
        gr_generate_summary.click(
            fn=gr_generate_summary_click,
            inputs=[gr_dataset, gr_selected_video, gr_full_video],
            outputs=[gr_summary_video, gr_visualization],
        )
    demo.launch(
        server_port=1234,
        ssl_verify=False,
    )
