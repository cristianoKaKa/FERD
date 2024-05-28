# Feature Enhancement with Reverse Distillation for Hyperspectral Anomaly Detection

## Overview

This repository contains the implementation of the algorithm described in the paper "Feature Enhancement with Reverse Distillation for Hyperspectral Anomaly Detection". 

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

