from keras.models import Sequential, Model
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal
from keras.applications import *
import keras.backend as K
import os
import time
from FCN8s_keras import FCN
import cv2
from tqdm import tqdm
model = FCN()
model.load_weights("Keras_FCN8s_face_seg_YuvalNirkin.h5")


def vgg_preprocess(im):
    im = cv2.resize(im, (500, 500))
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_[np.newaxis,:]
    #in_ = in_.transpose((2,0,1))
    return in_
  

DATA_DIR='/home/zhangao/projects/face-attack/dataset/securityAI_round1_images'
SEG_DIR = '/home/zhangao/projects/face-attack/dataset/securityAI_round1_mask'

if not os.path.exists(SEG_DIR):
    os.mkdir(SEG_DIR)
for filename in tqdm(os.listdir(DATA_DIR)):
    if filename[-4:]=='.jpg':
        fn = os.path.join(DATA_DIR,filename)
        im = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)
        inp_im = vgg_preprocess(im)
        out = model.predict([inp_im])
        out_resized = cv2.resize(np.squeeze(out), (im.shape[1],im.shape[0]))
        out_resized_clipped = np.clip(out_resized.argmax(axis=2), 0, 1).astype(np.float64)
        np.save(os.path.join(SEG_DIR,filename+'.npy'),out_resized_clipped)