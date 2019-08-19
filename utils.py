import os
import torch
from PIL import Image
import numpy as np
import random

def img2tensor(img,device):
    '''Normalize to [0,1]'''

    # handle numpy array
    tensor = np.zeros((1, 3, 112, 112), dtype=np.float32)
    tensor[0,:,:,:] = np.array(img.convert('RGB')).transpose(2,0,1)/255.0
    tensor = torch.from_numpy(tensor)
    tensor = tensor.to(device)
    return tensor


# Mean and Standard Deiation of the Dataset
mean = [0.5, 0.5, 0.5]
std = [0.5,0.5, 0.5]
def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]
    return t

def un_normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] * std[0]) + mean[0]
    t[:, 1, :, :] = (t[:, 1, :, :] * std[1]) + mean[1]
    t[:, 2, :, :] = (t[:, 2, :, :] * std[2]) + mean[2]
    return t

def l2_normlize(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

def extract_feature(img,model,device):
    return l2_normlize(model(normalize(img2tensor(img,device))))