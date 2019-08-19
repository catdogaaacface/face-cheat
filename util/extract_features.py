import os
import torch
import numpy as np

from PIL import Image

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
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

def generate_id_embeddings(multi_gpu, device, embedding_size, batch_size, backbone, carray):
    if multi_gpu:
        backbone = backbone.module # unpackage model from DataParallel
        backbone = backbone.to(device)
    else:
        backbone = backbone.to(device)
    backbone.eval() # switch to evaluation mode

    idx = 0
    embeddings = np.zeros([len(carray), embedding_size])
    print("Start generating identities embeddings!")
    with torch.no_grad():
        while idx + batch_size <= len(carray):
            
            batch = torch.tensor(carray[idx:idx + batch_size])

            embeddings[idx:idx + batch_size] = l2_normlize(backbone(normalize(batch.to(device)))).cpu()
            idx += batch_size
        if idx < len(carray):
            batch = torch.tensor(carray[idx:])
            embeddings[idx:] = l2_normlize(backbone(normalize(batch.to(device)))).cpu()
    print("Finish generating identities embeddings!")
    return embeddings

def distance_loss(id_embeddings,batch_embeddings,batch_targets):
    target_embeddings = torch.tensor(id_embeddings[batch_targets])
    loss = -torch.sum(torch.mul(target_embeddings,batch_embeddings))




