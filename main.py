import os
from glob import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import cv2 as cv
import matplotlib.pyplot as plt

from model.config import *
from model.ddpm_model import *
from data.CustomDataset import *
from run.run_script import *

if __name__ == '__main__':
    
    print('---------loading dataset---------')
    dataset = CustomImage(config.image_path)
    loader = Get_DataLoader(dataset, DIP_Method=True)
    print('---------finish loading----------')
    
    print('---------loading model-----------')
    print('without using time steps to noise')
    ddpm = DDPM(UNet(config.n_steps, DIP_Method=True), n_steps=config.n_steps, min_beta=config.min_beta, max_beta=config.max_beta, device=config.device, DIP_Method=True, time_step=False)
    optimizer=torch.optim.Adam(ddpm.parameters(), 0.002)
    train(ddpm, loader, config.epochs, optimizer, device=config.device, display=True)
    
    print('----------------------------------')
    print('using time steps to noise')
    ddpm = DDPM(UNet(config.n_steps, DIP_Method=True), n_steps=config.n_steps, min_beta=config.min_beta, max_beta=config.max_beta, device=config.device, DIP_Method=True, time_step=True)
    optimizer=torch.optim.Adam(ddpm.parameters(), 0.002)
    train(ddpm, loader, config.epochs, optimizer, device=config.device, display=True)