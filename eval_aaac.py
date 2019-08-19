import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import pickle
import os
from tqdm import tqdm
from utils import img2tensor, l2_normlize, normalize, extract_feature
from PIL import Image

from model import model,device

DATA_DIR = '/home/zhangao/projects/face-attack/dataset/securityAI_round1_images'
ADV_DIR ='/home/zhangao/projects/face-attack/dataset/pgd_Linf_eps12_iter10_ens4_mask'
THRESHOLD = 0.2

acc = 0
total=0

for filename in tqdm(os.listdir(DATA_DIR)):
    if filename[-4:]=='.jpg':
        gallery_img = Image.open(os.path.join(DATA_DIR,filename))
        probe_img = Image.open(os.path.join(ADV_DIR,filename))
        gallery_embd = extract_feature(gallery_img,model,device)
        probe_embd = extract_feature(probe_img,model,device)
        dist = torch.sum(torch.mul(gallery_embd,probe_embd)).detach().cpu().numpy()
        total += 1
        if dist > THRESHOLD:
            acc += 1

print (float(acc)/total)
