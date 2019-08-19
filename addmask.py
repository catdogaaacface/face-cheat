import os
import numpy as np 
from PIL import Image
from tqdm import tqdm

DATA_DIR='/home/zhangao/projects/face-attack/dataset/securityAI_round1_images'
ADV_DIR='/home/zhangao/projects/face-attack/dataset/pgd_Linf_eps12_iter10_ens4'
MASK_DIR='/home/zhangao/projects/face-attack/dataset/securityAI_round1_mask'
MIX_DIR = '/home/zhangao/projects/face-attack/dataset/pgd_Linf_eps12_iter10_ens4_mask'

if not os.path.exists(MIX_DIR):
    os.mkdir(MIX_DIR)

for filename in tqdm(os.listdir(DATA_DIR)):
    if filename[-4:]=='.jpg':
        clean_img = Image.open(os.path.join(DATA_DIR,filename))
        adv_img = Image.open(os.path.join(ADV_DIR,filename))
        mask = np.load(os.path.join(MASK_DIR,filename+'.npy'))
        mask = np.expand_dims(mask,2)
        mask = np.repeat(mask,3,axis=2)
        clean_arr = np.array(clean_img,dtype=np.float)
        adv_arr = np.array(adv_img,dtype=np.float)
        new_arr = clean_arr * (1-mask) + adv_arr * mask
        new_arr = np.uint8(new_arr)
        new_img = Image.fromarray(new_arr)
        new_img.save(os.path.join(MIX_DIR,filename))
        
        