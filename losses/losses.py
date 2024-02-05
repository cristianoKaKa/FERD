import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

def CosLoss(a, b):
    #mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1-cos_loss(a[item].reshape(a[item].shape[0],-1),
                                      b[item].reshape(b[item].shape[0],-1)))
    return loss


