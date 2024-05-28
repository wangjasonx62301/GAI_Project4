import os
from glob import glob
import math
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import cv2 as cv
import matplotlib.pyplot as plt
import random
from model.config import config

class CustomImage(Dataset):
    
    def __init__(self, img_folder, transform=None):
        super().__init__()
        self.img_folder = img_folder
        if transform == None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Lambda(lambda x: (x - 0.5) * 2)
            ])
        else : self.transform = transform
        self.image_paths = glob(os.path.join(img_folder, '*.jpg'))
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv.imread(img_path)
        image = cv.resize(image, (128, 128))
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        if self.transform != None:
            image = self.transform(image)
        return image
    
def Get_DataLoader(dataset, DIP_Method=False):
    
    if DIP_Method == True:
        rand = random.randint(0, len(dataset) - 1)
        dataset = dataset[rand].view(1, 1, 128, 128)
    
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)    
    return loader

def show_image(images, title=""):
    
    fig = plt.figure(figsize=(5, 5))
    
    if type(images) is torch.Tensor and len(images.shape) == 4:
        # torch transform return the shape CHW 
        images = rearrange(images, 'b c h w -> b h w c')
        images = images.detach().cpu().numpy()
        print(images.shape)
        
        idx = 0
        s = math.ceil(len(images) ** (1/2))
    
        for _ in range(s):
            for _ in range(s):
                fig.add_subplot(s, s, idx + 1)
                
                if idx < len(images):
                    plt.imshow(images[idx], cmap='gray')
                    idx += 1
        
    else: 
        images = rearrange(images, 'c h w -> h w c')
        plt.imshow(images, cmap='gray')
    
    fig.suptitle(title, fontsize=20)
    
    plt.show()
    
def show_first_batch(loader):
    for batch in loader:
        show_image(batch, 'Image in First batch')
        break
    
                
def generate_image(ddpm, n_samples=16, device=None, C=1, H=128, W=128):
    
    with torch.no_grad():
        if device is None:
            device = ddpm.device
            
        x = torch.randn(n_samples, C, H, W).to(device)
        
        if ddpm.DIP_Method == False:
        
            for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
                            
                time_T = (torch.ones(n_samples, 1) * t).to(device).long()
                eta_theta = ddpm.backward(x, time_T)
                
                alpha_t = ddpm.alphas[t]
                alpha_t_bar = ddpm.alpha_bars[t]
                
                # (1 / a^2) * (x - (1 - a) / (1 - a_hat)^2 * eta_t)
                x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)
                
                
                # print('after x', x.mean(), x.var())
                
                if t > 0:
                    z = torch.randn(n_samples, C, H, W).to(device)
                    
                    beta_t = ddpm.betas[t]
                    sigma_t = beta_t.sqrt()

                    x = x + sigma_t * z
                    
                    # print('after sigma', x.mean(), x.var())
                    
        else :
            
            x = torch.randn(1, C, H, W).to(device)
            for _ in range(n_samples - 1):
                x = torch.cat((x, torch.randn(1, C, H, W).to(device)), dim=0)

            x = ddpm.backward(x, None)
                
        return x

        