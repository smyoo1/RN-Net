# RN-Net

# Introduction

RN-Net is a neural network deploy short-term memory memristor's device to work with the event-based camera.
This repository provides an implementation of RN-Net that is compatible with a Python package, snnTorch.

# Demonstration

5 different tasks (CIFAR10-DVS, NCaltech 101, N-CARS, DVS128 Gesture, and DVS Lip) are demonstrated by RN-Net in the paper. Codes for the five datasets can be found in this repository. Links for the pre-processed data (output of $R_{in}$) will be updated here soon.

# Citation
If you use this repo in your work, please cite our preprint:
```bib
@article{https://doi.org/10.1002/aisy.202400265,
author = {Yoo, Sangmnin and Lee, Eric Yeu-Jer and Wang, Ziyu and Wang, Xinxin and Lu, Wei D.},
title = {RN-Net: Reservoir Nodes-Enabled Neuromorphic Vision Sensing Network},
journal = {Advanced Intelligent Systems},
volume = {n/a},
number = {n/a},
pages = {2400265},
keywords = {event-based camera, memristor, neuromorphic, reservoir computing, SNN},
doi = {https://doi.org/10.1002/aisy.202400265},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/aisy.202400265},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/aisy.202400265},
abstract = {Neuromorphic computing systems promise high energy efficiency and low latency. In particular, when integrated with neuromorphic sensors, they can be used to produce intelligent systems for a broad range of applications. An event-based camera is such a neuromorphic sensor, inspired by the sparse and asynchronous spike representation of the biological visual system. However, processing the event data requires either using expensive feature descriptors to transform spikes into frames, or using spiking neural networks (SNNs) that are expensive to train. In this work, a neural network architecture is proposed, reservoir nodes-enabled neuromorphic vision sensing network (RN-Net), based on dynamic temporal encoding by on-sensor reservoirs and simple deep neural network (DNN) blocks. The reservoir nodes enable efficient temporal processing of asynchronous events by leveraging the native dynamics of the node devices, while the DNN blocks enable spatial feature processing. Combining these blocks in a hierarchical structure, the RN-Net offers efficient processing for both local and global spatiotemporal features. RN-Net executes dynamic vision tasks created by event-based cameras at the highest accuracy reported to date at one order of magnitude smaller network size. The use of simple DNN and standard backpropagation-based training rules further reduces implementation and training costs.}
}
```
