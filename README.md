# MonoDFNet
<p align="center">MonoDFNet: Monocular 3D Object Detection with Depth Fusion and Adaptive Optimization</p>
MonoDFNet is a monocular 3D object detection network based on improvements to MonoCD. This repository provides the source code, datasets, and instructions to reproduce the results from the paper MonoDFNet: Monocular 3D Object Detection with Depth Fusion and Adaptive Optimization.
# Overview
Monocular 3D object detection offers a low-cost solution for 3D perception using a single camera. However, it faces challenges like occlusion, truncation, and the lack of depth information, which reduce detection accuracy in complex scenes. MonoDFNet addresses these challenges through:  

1.Parameter Adjustments: Optimizes the learning process to improve detection robustness and accuracy.< br >
2.Adaptive Focus Mechanism: Dynamically adjusts attention to important regions, reducing interference from irrelevant areas.< br >
3.Multi-Branch Depth Prediction with Weight Sharing: Efficiently integrates multi-scale depth information while maintaining computational efficiency.< br >
