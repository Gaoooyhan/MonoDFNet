# MonoDFNet
<p align="center">:blush:MonoDFNet: Monocular 3D Object Detection with Depth Fusion and Adaptive Optimization:blush:</p>
MonoDFNet is a monocular 3D object detection network based on improvements to MonoCD. This repository provides the source code, datasets, and instructions to reproduce the results from the paper MonoDFNet: Monocular 3D Object Detection with Depth Fusion and Adaptive Optimization.

# Overview
Monocular 3D object detection offers a low-cost solution for 3D perception using a single camera. However, it faces challenges like occlusion, truncation, and the lack of depth information, which reduce detection accuracy in complex scenes. MonoDFNet addresses these challenges through:  
1.Parameter Adjustments: Optimizes the learning process to improve detection robustness and accuracy.  

2.Adaptive Focus Mechanism: Dynamically adjusts attention to important regions, reducing interference from irrelevant areas.  

3.Multi-Branch Depth Prediction with Weight Sharing: Efficiently integrates multi-scale depth information while maintaining computational efficiency.  

# Key Features
(1)Accurate Depth Estimation: Enhanced depth prediction through multi-branch architecture and weight sharing.  

(2)Robustness in Complex Scenarios: Improved detection performance under occlusion, truncation, and cluttered environments.  

(3)Efficient Computation: Balanced performance and computational cost with lightweight module designs.  

# Installation
```python
git clone https://github.com/Gaoooyhan/MonoDFNet.git

cd MonoDFNet

conda create -n monocd python=3.7

conda activate monocd

conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch

pip install -r requirements.txt

cd model/backbone/DCNv2

sh make.sh

cd ../../..

python setup.py develop
```

# Data Preparation
Please download KITTI dataset and organize the data as follows:
```python
#ROOT		
  |training/
    |calib/
    |image_2/
    |label/
    |planes/
    |ImageSets/
  |testing/
    |calib/
    |image_2/
    |ImageSets/
```
The road planes for Horizon Heatmap training could be downloaded from MonoCD. Then remember to set the DATA_DIR = "/path/to/your/kitti/" in the config/paths_catalog.py according to your data path.

# Get Started
## Train
Training with one GPU.  
```python
CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py --batch_size 8 --config runs/monocd.yaml --output output/exp
```
## Test
The model will be evaluated periodically during training and you can also evaluate an already trained checkpoint with
```python
CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py --config runs/monocd.yaml --ckpt YOUR_CKPT  --eval
```
## Visualization
To visualize the detection results:
```python
CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py --config runs/monocd.yaml --ckpt YOUR_CKPT  --eval --vis_all
```
# Acknowledgement
This project benefits from awesome works of MonoCD.




