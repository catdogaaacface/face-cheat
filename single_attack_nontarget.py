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

def attack(model,  probe, gallery, issame, eps, attack_type, iters):
    adv = probe.detach()
    adv.requires_grad = True
    

    gallery_embd = l2_normlize(model(normalize(gallery.detach())))
    gallery_embd = gallery_embd.detach()
    if issame:
        flag = 1
    else: 
        flag = -1
    
    if attack_type == 'fgsm':
        iterations = 1
    else:
        iterations = iters

    if attack_type == 'pgd':
        step = 2 / 255
    else:
        step = eps / iterations  
        noise = 0
        
    for j in range(iterations):
        features = l2_normlize(model(normalize(adv.clone())))
        loss = -torch.sum(torch.mul(features,gallery_embd))
        loss.backward()
        if attack_type == 'mim':
            adv_mean= torch.mean(torch.abs(adv.grad), dim=1,  keepdim=True)
            adv_mean= torch.mean(torch.abs(adv_mean), dim=2,  keepdim=True)
            adv_mean= torch.mean(torch.abs(adv_mean), dim=3,  keepdim=True)
            adv.grad = adv.grad / adv_mean
            noise = noise + adv.grad
        else:
            noise = adv.grad
        # Optimization step
        adv.data = adv.data + flag * step * noise.sign()
        if attack_type == 'pgd':
            adv.data = torch.where(adv.data > probe.data + eps, probe.data + eps, adv.data)
            adv.data = torch.where(adv.data < probe.data - eps, probe.data - eps, adv.data)
        adv.data.clamp_(0.0, 1.0)
        adv.grad.data.zero_()
    return adv.detach()



DATA_DIR = 'aaac/lfwaaac'
ADV_DIR ='aaac/lfwadv'
THRESHOLD = 0.2

if not os.path.exists(ADV_DIR):
    os.mkdir(ADV_DIR)

for filename in tqdm(os.listdir(DATA_DIR)):
    if filename[-4:]=='.jpg':
        gallery_img = Image.open(os.path.join(DATA_DIR,filename))
        probe_img = Image.open(os.path.join(DATA_DIR,filename))
        gallery_tensor = img2tensor(gallery_img,device)
        probe_tensor = img2tensor(probe_img,device)
        adv = attack(model,probe_tensor,gallery_tensor,True,eps=8.0/255,attack_type='pgd',iters=10)
        adv_arr = adv.cpu().numpy().squeeze().transpose(1,2,0)*255
        adv_arr = adv_arr.astype(np.uint8)
        adv_img = Image.fromarray(adv_arr)
        adv_img.save(os.path.join(ADV_DIR,filename))

    # for j in tqdm(range(num_pairs)):
    #     line = lines[line_num].strip()
    #     line = line.split()
    #     line_num += 1
    #     gallery_name, gallery_idx, probe_name, probe_idx = line[0],int(line[1]),line[2],int(line[3])
    #     gallery_img = Image.open(os.path.join(GALLERY_DIR,gallery_name,gallery_name+'_%04d.jpg'%(gallery_idx)))
    #     probe_img = Image.open(os.path.join(PROBE_DIR,probe_name,probe_name+'_%04d.jpg'%(probe_idx)))
    #     gallery_tensor = img2tensor(gallery_img,device)
    #     probe_tensor = img2tensor(probe_img,device)
    #     adv = attack(model,probe_tensor,gallery_tensor,False,eps=8.0/255,attack_type='pgd',iters=10)
    #     adv_arr = adv.cpu().numpy().squeeze().transpose(1,2,0)*255
    #     adv_arr = adv_arr.astype(np.uint8)
    #     adv_img = Image.fromarray(adv_arr)
    #     adv_img.save(os.path.join(diffdir,probe_name+'_%04d_%s_%04d.jpg'%(probe_idx,gallery_name,gallery_idx)))
