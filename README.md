# Feature Enhancement with Reverse Distillation for Hyperspectral Anomaly Detection
**Authors**: Wenping Jin, Feng Dang, Li Zhu

## Overview

This repository contains the implementation of the algorithm described in the paper "Feature Enhancement with Reverse Distillation for Hyperspectral Anomaly Detection". 

## Abstract
Most hyperspectral anomaly detection methods based on trainable parameter networks require parameter adjustments or retraining on new test scenes, resulting in high time consumption and unstable performance, limiting their practicality in large-scale anomaly detection scenarios. In this letter, we address this issue by proposing a novel Feature Enhancement Network (FEN). FEN does not directly compute anomaly scores but enhances the background features of the original hyperspectral image, thereby improving the performance of non-training-based anomaly detection algorithms, such as those based on Mahalanobis distance. To achieve this, FEN needs to be trained on a background dataset. Once trained, it does not require retraining for new detection scenes, significantly reducing time costs. Specifically, during training on background data, a complete network is trained using a reverse distillation framework with a spectral feature alignment mechanism to enhance the network's ability to express background features. For inference, a pruned version of this network, which is FEN, is applied, consisting solely of components most relevant to expressing features in the spectral dimension. This design effectively reduces redundant information, enhancing both inference efficiency and anomaly detection accuracy. Experimental results demonstrate that our method can significantly improve the performance of Mahalanobis distance-based anomaly detection methods while incurring minimal time costs.

![image](https://github.com/cristianoKaKa/FERD/blob/master/framework.png)

## Usage

### Dataset

HAD100: https://zhaoxuli123.github.io/HAD100/

### Environment

- torch
- torchvision
- numpy
- scikit-learn

### Running Tests

To run the tests on HAD100, use the following command:

- For first 50 bands test:
```bash
python test.py --input_channel 50
```
- For first 100 bands test:
```bash
python test.py --input_channel 100
```
- For first 200 bands test:
```bash
python test.py --input_channel 200
```

### Running Training on HAD100

```bash
python train.py
```

