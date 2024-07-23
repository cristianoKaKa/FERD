# Feature Enhancement with Reverse Distillation for Hyperspectral Anomaly Detection
**Authors**: Wenping Jin, Feng Dang, Li Zhu

## Overview

This repository contains the implementation of the algorithm described in the paper "Feature Enhancement with Reverse Distillation for Hyperspectral Anomaly Detection". 

## Abstract
Most hyperspectral anomaly detection methods based on trainable parameter networks require parameter adjustments or retraining on new test scenes, resulting in high time consumption and unstable performance, limiting their practicality in large-scale anomaly detection scenarios. In this letter, we address this issue by proposing a novel Feature Enhancement Network (FEN). FEN does not directly compute anomaly scores but enhances the background features of the original hyperspectral image, thereby improving the performance of non-training-based anomaly detection algorithms, such as those based on Mahalanobis distance. To achieve this, FEN needs to be trained on a background dataset. Once trained, it does not require retraining for new detection scenes, significantly reducing time costs. Specifically, during training on background data, a complete network is trained using a reverse distillation framework with a spectral feature alignment mechanism to enhance the network's ability to express background features. For inference, a pruned version of this network, which is FEN, is applied, consisting solely of components most relevant to expressing features in the spectral dimension. This design effectively reduces redundant information, enhancing both inference efficiency and anomaly detection accuracy. Experimental results demonstrate that our method can significantly improve the performance of Mahalanobis distance-based anomaly detection methods while incurring minimal time costs.

![image](https://github.com/cristianoKaKa/FERD/blob/master/framework.png)

## Usage on HAD100 Dataset

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
### Results

| Bands | mAUC |
| :--: | :--: |
| The First 50 Bands | 0.9941 |
| The First 100 Bands | 0.9901 |
| The First 200 Bands | 0.9890 |

## Feature Enhancement Performance on Other Real-World Scenes

|     Scenes    | GRX AUC | GRX Time | FEN+GRX AUC | FEN+GRX Time | SLRX AUC | SLRX Time | FEN+SLRX AUC | FEN+SLRX Time |
|---------------|---------|----------|-------------|--------------|----------|-----------|--------------|---------------|
| Abu-airport-1 | 0.8248  | 0.13     | **0.9478**  | 0.29         | 0.9471   | 1.79      | **0.9653**   | 1.93          |
| Abu-airport-2 | 0.8433  | 0.13     | **0.9861**  | 0.29         | 0.9527   | 1.79      | **0.9869**   | 1.93          |
|   Tularosa    | 0.9126  | 0.43     | **0.9868**  | 0.60         | 0.9396   | 6.51      | **0.9723**   | 6.73          |

The experimental code is located in the Supp_Exp folder, which includes the code for FERD as well as GRX and SLRX. SLRX is a more efficient Local RX algorithm designed by us, which uses a fixed patch instead of the traditional sliding window in Local RX algorithms.

The trained model in TorchScript format can be found at [https://huggingface.co/xjpha/FERD](https://huggingface.co/xjpha/FERD). This model can be used for anomaly detection feature enhancement in any scenario with just a few lines of code:

```python
import torch

# If it is an HSI of 200 bands captured by AVIRIS:
hsi = torch.rand(1, 200, 100, 100)

# Load models:
cwl = torch.jit.load('aviris/200bands/cwl_script.pt')
rnl = torch.jit.load('aviris/200bands/rnl_script.pt')
pfl = torch.jit.load('aviris/200bands/fpl_script.pt')
cwl.eval()
rnl.eval()
pfl.eval()

# The enhanced HSI can be computed:
enhanced_hsi, _ = pfl(rnl(cwl(hsi)), hsi)
```

## Citation
If this repo works positively for your research, please consider citing our paper. Thanks all!
```
@article{jin2024feature,
  title={Feature Enhancement with Reverse Distillation for Hyperspectral Anomaly Detection},
  author={Jin, Wenping and Dang, Feng and Zhu, Li},
  journal={Authorea Preprints},
  year={2024},
  publisher={Authorea}
}
```
