# Lidar Stereo Fusion
This repository is a research and in-progress project to explore lidar and stereo fusion strategies. The main papers that have been surveyed are in the `papers` folder.
Currently, this repository implements the following paper (but with only truncated L2 Lidar loss):

Xuelian Cheng*, Yiran Zhong*, Yuchao Dai, Pan Ji, Hongdong Li.
_Noise-Aware Unsupervised Deep Lidar-Stereo Fusion._
In CVPR, 2019. https://arxiv.org/abs/1904.03868.

At the time of publishing of our project, the original author's repo hasn't been updated to contain their source code. However, you may
monitor their repository for real-time updates in the future: https://github.com/AvrilCheng/LidarStereoNet/

Resources used in implementing the LidarStereoNet network architecture:

PSMNet source code at: https://github.com/JiaRenChang/PSMNet
DepthComplete project at: https://github.com/yxgeee/DepthComplete

## Navigating the Repo
The `main.py` script is the main script for training LidarStereoNet on a GPU-enabled instance (in this project's case, a K80 GPU was used on Google Cloud). The `utils` folder contains all the utility scripts that were used for batch data processing. The `models` folder contains Python files that define the entire LidarStereoNet architecture. The `kitti_loader.py` file is the dataloader to use for training. The `kitti_loader_eval.py` is the dataloader to use for testing/eval. Also, `eval.py` takes a trained PyTorch model and evaluates it on KITTI SceneFlow 2015's metrics. The `example.py` file helps you visualize a trained LidarStereoNet model's output disparity image. The `slow_naive_fusion.py` file implements a naive fusion strategy that doesn't rely on deep learning or probabilistic models. This strategy, which is a little slow, uses Lidar measurements as the true disparity for all pixels which have a corresponding Lidar point. For pixels that don't have a Lidar measurement, it takes a 6px by 6px window around that pixel and averages the median Lidar measurements of all valid pixels in the window with the stereo disparity calculation of that pixel. If neither is available, it sets disparity to 0.

## Future Work
Future work should focus on accelerating the naive fusion approach by exploiting vectorized operations in numpy. Also, all four loss components as detailed in the original LidarStereoNet paper should be implemented to achieve good results.
