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
from model.ddpm_model import *
from data.CustomDataset import *

def eval_lr(ddpms, learning_rate, origin_image, average_dict):
    
    def cmp(s, dt, t, lr, device):
        dt, t = dt.to(device), t.to(device)
        ex = torch.all(dt == t).item()
        app = torch.allclose(dt, t)
        maxdiff = (dt - t).abs().max().item()
        average_dict[lr].append(maxdiff)
        print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')
    
    gen_images = None
    for ddpm, lr in zip(ddpms, learning_rate):
        gen = generate_image(ddpm, n_samples=1)
        if gen_images == None: gen_images = gen
        else: gen_images = torch.cat((gen_images, gen), dim=0)
        cmp(f'using lr : {lr:.5f}', origin_image, gen, lr, ddpm.device)
        
    show_image(gen_images)
    

def train(ddpm, loader, epochs, optim, device, display=False):
    criterion = nn.L1Loss()
    n_steps = ddpm.n_steps
    losses = []
    gen_images = None
    for epoch in range(epochs):
        cur_loss = 0.0
        for step, batch in enumerate(loader):
            
            x0 = batch.to(device)
            if ddpm.DIP_Method == False:
                
                n = len(x0)
                eta = torch.randn_like(x0).to(device)
                t = torch.randint(0, n_steps, (n, )).to(device).view(n, -1)
                noise = ddpm(x0, t[0], eta)
                eta_theta = ddpm.backward(noise, t)
                
                loss = criterion(eta_theta, eta)
            
            else :
                n = len(x0)
                t = torch.randint(0, n_steps, (n, )).to(device).view(n, -1)
                noise = ddpm(x0, t[0]).to(device)
                # noise = ddpm(x0, None).to(device)
                gen = ddpm.backward(noise, None).to(device)
                loss = criterion(gen, x0)
                
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            cur_loss += loss.item() * len(x0) / len(loader.dataset)

        losses.append(cur_loss)      
        if epoch % 10 == 0:     
            print(f'{epoch:3d} / {epochs:3d} loss : {cur_loss:.5f}')
            
        if display == True and epoch % 50 == 0 :
            n_samples = 16 if ddpm.DIP_Method == False else 1
            generated = generate_image(ddpm, n_samples=n_samples)
            if ddpm.DIP_Method == False:
                show_image(generated)
            else :
                if gen_images == None : gen_images = generated
                else : gen_images = torch.cat((gen_images, generated), dim=0)
        
    
    plt.plot(losses)
    
    if display == True:
        if ddpm.DIP_Method == False:
            n_samples = 16
            generated = generate_image(ddpm, n_samples=n_samples)
            show_image(generated)
            
        else: show_image(gen_images)

    
    
        
