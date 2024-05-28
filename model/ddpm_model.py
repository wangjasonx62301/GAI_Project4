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
from model.config import config

class DDPM(nn.Module):
    
    def __init__(self, backward_=None, n_steps=100, device=None, min_beta=0.0001, max_beta=0.02, DIP_Method=False, time_step=True):
        super().__init__()
        self.image_shape = (1, 128, 128)
        if backward_ != None:
            self.backward_ = backward_.to(device)
        self.n_steps = n_steps
        self.device = device
        # get the tensor start with min_beta to max_beta with n_steps, [0.0001, ....., 0.02]
        self.betas = torch.linspace(min_beta, max_beta, self.n_steps).to(device)
        self.alphas = 1 - self.betas
        # alpha_bar_t = alpha_t * alpha_t-1 * ..... * alpha_1
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)
        self.DIP_Method = DIP_Method
        self.time_step = time_step
        
    def forward(self, x, t, eta=None):
        B, C, H, W = x.shape                                                    # batch_size, channel, height, width
        
        if self.time_step == True:
            
            alpha_bar = torch.tensor([self.alpha_bars[t]])                      # get current alpha_bar by current time_step
            alpha_bar = repeat(alpha_bar, 'C -> B C', B=B).to(self.device)      # repeat it B times(for batch calculation)
            
            # eta is a guassian distribution
            if eta == None:
                eta = torch.randn(B, C, H, W).to(self.device)
            
            # q = alpha_hat.sqrt() * x_0 + (1 - alpha_hat).sqrt() * eta
            noise = alpha_bar.sqrt().view(B, 1, 1, 1) * x.to(self.device) + (1 - alpha_bar).sqrt().view(B, 1, 1, 1) * eta
            
        else :
            noise = torch.randn(B, C, H, W)    
        
        return noise
    
    def backward(self, x, t):
        # this is not the actual backpropagation, in DDPM we have forward(make noise) and backward(predict from noise)
        return self.backward_(x, t)

        
class Block(nn.Module):
    
    def __init__(self, shape, input_channel, output_channel, kernel_size=3, stride=1, padding=1, activation=True, normalize=True):
        super().__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is not None else nn.Identity()
        self.norm = normalize
        
    def forward(self, x):
        out = self.ln(x) if self.norm == True else x
        out = self.activation(self.conv1(out))
        out = self.activation(self.conv2(out))
        return out


class SinusoidalPosEmb(nn.Module):
    def __init__(self, n, n_embd, theta=10000):
        super().__init__()
        self.emb_wei = torch.zeros(n, n_embd)
        wei = torch.tensor([1 / theta ** (2 * j / n_embd) for j in range(n_embd)]).view(1, n_embd)
        t = torch.arange(n).view(n, 1)
        # even idx embedding
        self.emb_wei[:, ::2] = torch.sin(t * wei[:, ::2])
        self.emb_wei[:, 1::2] = torch.cos(t * wei[:, ::2])
        
        self.embedding = nn.Embedding(n, n_embd)
        self.embedding.weight.data = self.emb_wei
        
    def forward(self, x):
        out = self.embedding(x)
        return out

class mlp(nn.Module):
    
    def __init__(self, f_in, f_out):
        super().__init__()
        self.mlp_f = nn.Sequential(
            nn.Linear(f_in, f_out),
            nn.SiLU(),
            nn.Linear(f_out, f_out),
        )
        
    def forward(self, x):
        out = self.mlp_f(x)
        return out
    
class UNet(nn.Module):
    
    def __init__(self, n_steps=1000, time_n_embd=100, DIP_Method=False):
        super().__init__()
        self.DIP_Method = DIP_Method
        # positional time embedding
        self.time_embd = SinusoidalPosEmb(n_steps, time_n_embd)
        
        # Down sampling
        self.time_emb1 = mlp(time_n_embd, 1)
        self.block1 = nn.Sequential(
            Block((1, 128, 128), 1, 10),
            Block((10, 128, 128), 10, 10),
            Block((10, 128, 128), 10, 10),
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)
        
        self.time_emb2 = mlp(time_n_embd, 10)
        self.block2 = nn.Sequential(
            Block((10, 64, 64), 10, 20),
            Block((20, 64, 64), 20, 20),
            Block((20, 64, 64), 20, 20),
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)
        
        self.time_emb3 = mlp(time_n_embd, 20)
        self.block3 = nn.Sequential(
            Block((20, 32, 32), 20, 40),
            Block((40, 32, 32), 40, 40),
            Block((40, 32, 32), 40, 40),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(40, 40, 3, 2, 1),
            nn.SiLU(),
        )

        # Bottleneck
        self.time_emb_mid = mlp(time_n_embd, 40)
        self.block_mid = nn.Sequential(
            Block((40, 16, 16), 40, 20),
            Block((20, 16, 16), 20, 20),
            Block((20, 16, 16), 20, 40),
        )
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(40, 40, 3, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(40, 40, 2, 1),
        )
        
        self.time_emb4 = mlp(time_n_embd, 80)
        self.block4 = nn.Sequential(
            Block((80, 32, 32), 80, 40),
            Block((40, 32, 32), 40, 20),
            Block((20, 32, 32), 20, 20),
        )
        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        
        self.time_emb5 = mlp(time_n_embd, 40)
        self.block5 = nn.Sequential(
            Block((40, 64, 64), 40, 20),
            Block((20, 64, 64), 20, 10),
            Block((10, 64, 64), 10, 10),
        )
        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        
        self.time_emb_out = mlp(time_n_embd, 20)
        self.block_out = nn.Sequential(
            Block((20, 128, 128), 20, 10),
            Block((10, 128, 128), 10, 10),
            Block((10, 128, 128), 10, 10, normalize=False),
        )
        
        self.conv_out = nn.Conv2d(10, 1, 3, 1, 1)
    
    def forward(self, x, t):
        
        if self.DIP_Method == False:
            
            t = self.time_embd(t)
            N = len(x)
            out1 = self.block1(x + self.time_emb1(t).view(N, -1, 1, 1))
            out2 = self.block2(self.down1(out1) + self.time_emb2(t).view(N, -1, 1, 1))
            out3 = self.block3(self.down2(out2) + self.time_emb3(t).view(N, -1, 1, 1))
            
            out_mid = self.block_mid(self.down3(out3) + self.time_emb_mid(t).view(N, -1, 1, 1))

            out4 = torch.cat((out3, self.up1(out_mid)), dim=1)
            out4 = self.block4(out4 + self.time_emb4(t).view(N, -1, 1, 1))
            
            out5 = torch.cat((out2, self.up2(out4)), dim=1)
            out5 = self.block5(out5 + self.time_emb5(t).view(N, -1, 1, 1))
            
            out = torch.cat((out1, self.up3(out5)), dim=1)
            out = self.block_out(out + self.time_emb_out(t).view(N, -1, 1, 1))
            
            out = self.conv_out(out)
        
        else :
            # without time embedding
            out1 = self.block1(x)
            out2 = self.block2(self.down1(out1))
            out3 = self.block3(self.down2(out2))
            
            out_mid = self.block_mid(self.down3(out3))

            out4 = torch.cat((out3, self.up1(out_mid)), dim=1)
            out4 = self.block4(out4)
            
            out5 = torch.cat((out2, self.up2(out4)), dim=1)
            out5 = self.block5(out5)
            
            out = torch.cat((out1, self.up3(out5)), dim=1)
            out = self.block_out(out)
            
            out = self.conv_out(out)
        
        return out

# ddpm = DDPM(UNet(config.n_steps, DIP_Method=False), n_steps=config.n_steps, min_beta=config.min_beta, max_beta=config.max_beta, device=config.device, DIP_Method=False)
# print('test')