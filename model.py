from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import argparse


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()


    def forward(self, x):
 

def convfc():
    model_arch = Net()
    return model_arch