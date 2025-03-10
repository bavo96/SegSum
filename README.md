# SegSum
PyTorch Implementation of SegSum model in the paper **Integrate the temporal scheme for unsupervised video summarization via attention mechanism** [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10904447)] [[DOI](https://doi.org/10.1109/ACCESS.2025.3546149)] [[Cite](https://github.com/bavo96/SegSum#citation)]

# Prerequisites
- Python 3.10 \
- Install requirements ```pip install -r requirements.txt```

# Data
Pretrained models and training/test data are available [here](https://drive.google.com/drive/folders/1IXWNZTc2LbIPmhR7VpysDHO-LaCjHesg?usp=sharing). Extract TVSum/SumMe rar files into [./data/SumMe](./data/SumMe) and [./data/TVSum](./data/TVSum). The `pickle` files are the features of the video frames extracted using GoogleNet and the `h5` files were obtained from Kaiyang Zhou.

## pickle file
```
/key
    /features                 2D-array with shape (num_video_frames, feature-dimension)
```
## h5 file
```Text
/key
    /features                 2D-array with shape (n_steps, feature-dimension)
    /gtscore                  1D-array with shape (n_steps), stores ground truth importance score (used for training, e.g. regression loss)
    /user_summary             2D-array with shape (num_users, n_frames), each row is a binary vector (used for test)
    /change_points            2D-array with shape (num_segments, 2), each row stores indices of a segment
    /n_frame_per_seg          1D-array with shape (num_segments), indicates number of frames in each segment
    /n_frames                 number of frames in original video
    /picks                    positions of sub-sampled frames in original video
    /n_steps                  number of sub-sampled frames
    /gtsummary                1D-array with shape (n_steps), ground truth summary provided by user (used for training, e.g. maximum likelihood)
    /video_name (optional)    original video name, only available for SumMe dataset
```

# Training
To reproduce our training, run ```python -m train.search_params```

# Evaluation
To evaluate our best models, put the `best_models` in `inference` folder and run `python inference/inference_data.py`

# Citation

If you find our work, code or pretrained models, useful in your work, please cite the following publication:

E. Apostolidis, G. Balaouras, V. Mezaris, I. Patras, "Summarizing Videos using Concentrated Attention and Considering the Uniqueness and Diversity of the Video Frames", Proc. of the 2022 Int. Conf. on Multimedia Retrieval (ICMR ’22), June 2022, Newark, NJ, USA.

```
@ARTICLE{10904447,
  author={Bang, Vo Quoc and Viet, Vo Hoai},
  journal={IEEE Access}, 
  title={Integrate the Temporal Scheme for Unsupervised Video Summarization via Attention Mechanism}, 
  year={2025},
  volume={13},
  number={},
  pages={38147-38162},
  keywords={Deep learning;Training;Attention mechanisms;Data models;Convolutional neural networks;Computer architecture;Transformers;Neurons;Kernel;Brain modeling;Video summarization;unsupervised learning;temporal video segmentation},
  doi={10.1109/ACCESS.2025.3546149}}
```

## Acknowledgement
This repo is built mainly on the CA-SUM repo: (https://github.com/e-apostolidis/CA-SUM).

We thank all of the authors of this repo for their contributions.
