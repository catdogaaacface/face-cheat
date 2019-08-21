import torch
import os
from tqdm import tqdm
from utils import img2tensor, l2_normlize, normalize, extract_feature
from PIL import Image
import numpy as np

from model import model, device

DATA_DIR = 'data/securityAI_round1_images'
CSV_PATH = 'data/securityAI_round2_dev.csv'
ADV_DIR = 'data/adv_pgd'
# ADV_DIR = DATA_DIR
THRESHOLD = 0.2

acc = 0
total = 0

# Extract all gallery features
with open(CSV_PATH) as f:
    lines = f.readlines()[1:]
filenames = []
person_names = []
person_ids = []
for line in lines:
    person_id, filename, person_name = line.split(',')
    person_id = int(person_id) - 1
    person_ids.append(person_id)
    filenames.append(filename)
    person_names.append(person_name)
gallery_features = torch.zeros((712, 512), dtype=torch.float32).detach().to(device)
print("Extracting gallery features...")
for i in tqdm(range(len(filenames))):
    img = Image.open(os.path.join(DATA_DIR, filenames[i]))
    embd = extract_feature(img, model, device).detach()
    gallery_features[i, :] = embd[0, :]

l2_dists = []
acc = 0.

def l2distance(image1, image2):
    """
    Compute L2 Norm of two images.
    :param image1: original image
    :param image2: adversarial image
    :return: L2 distance
    """
    image1 = np.array(image1).astype(np.float)
    image2 = np.array(image2).astype(np.float)
    image2 = np.clip(image2, image1-25.5, image1+25.5)
    diff = image1 - image2
    return np.sqrt(np.multiply(diff, diff).sum(2)).mean()


for i in range(len(filenames)):
    probe = Image.open(os.path.join(ADV_DIR, filenames[i]))
    gallery = Image.open(os.path.join(DATA_DIR, filenames[i]))
    l2 = l2distance(gallery, probe)
    probe_embd = extract_feature(probe, model, device)
    sim = torch.matmul(gallery_features, probe_embd.t()).detach().cpu()
    pred_label = sim.argmax(dim=0).item()
    pred_sim = sim[pred_label, 0]
    print("Predicted label: {}, true label: {}, similarity: {:.2f}, l2 distance: {:.2f}".format(
        pred_label, person_ids[i], pred_sim.item(), l2))
    if not (pred_label == person_ids[i] and pred_sim > THRESHOLD):
        l2_dists.append(l2)
    else:
        acc += 1
        l2_dists.append(44.1673)
print("Total accuracy: {:.4f}, L2 distance: {:.4f}".format(acc / len(filenames), np.mean(l2_dists)))


