# SegSum
PyTorch Implementation of ***SegSum*** model in the paper ***Integrate the temporal scheme for unsupervised video summarization via attention mechanism*** [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10904447)] [[DOI](https://doi.org/10.1109/ACCESS.2025.3546149)] [[Cite](https://github.com/bavo96/SegSum#citation)]

## I. Prerequisites
- Python 3.10 
- Install requirements ```pip install -r requirements.txt```

## II. Data
Pretrained models and training/test data are available [here](https://drive.google.com/drive/folders/1IXWNZTc2LbIPmhR7VpysDHO-LaCjHesg?usp=sharing). This link has 2 folders:
### Features
- Including the `pickle` files are the features of the video frames extracted using GoogleNet and the `h5` files were obtained from this repo [DR-DSN](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce) by Kaiyang Zhou. Below are details of the files:
    - pickle file
    ```
    /key
        /features                 2D-array with shape (num_video_frames, feature-dimension)
    ```
    
    - h5 file
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
### Models
- Including the best models in the paper.

For training and evaluation, extract TVSum/SumMe rar files into [./data/SumMe](./data/SumMe) and [./data/TVSum](./data/TVSum). 

## III. Training
To reproduce our training, run 
```bash
python -m train.search_params
```

## IV. Evaluation
To evaluate our best models, put the `best_models` in `inference` folder and run 
```bash 
python inference/inference_data.py
```

## V. Demo
To run our demo, run 
```bash
python app/app.py
```

## VI. Citation

If our work, code, or pretrained models contribute to your research or projects, please cite the following publication:

Vo, B. Q., & Vo, V. H., "Integrate the Temporal Scheme for Unsupervised Video Summarization via Attention Mechanism," in IEEE Access, vol. 13, pp. 38147-38162, 2025, doi: 10.1109/ACCESS.2025.3546149.

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

## VII. Acknowledgement
This repo is built mainly on the CA-SUM repo: (https://github.com/e-apostolidis/CA-SUM).

We thank all of the authors of this repo for their contributions.
