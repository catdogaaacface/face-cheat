from PIL import Image
import torch
import torch.nn as nn

import os

from backbone.model_irse import IR_50 , IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
CKPT_LIST =['backbone_resnet50.pth','backbone_ir50_ms1m.pth','backbone_ir50_asia.pth','backbone_ir152.pth']
CKPT_LIST = [os.path.join('/home/zhangao/model_zoo/face_model_zoo',ckpt) for ckpt in CKPT_LIST]

model_list = [ResNet_50([112,112]),IR_50([112,112]),IR_50([112,112]),IR_152([112,112])]

model_index = 1

MULTIGPU=False
GPU_ID=''

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

#model = IR_50([112,112])
#model = ResNet_50([112,112])
#model = IR_SE_50([112,112])

model = model_list[model_index]
model.load_state_dict(torch.load(CKPT_LIST[model_index],map_location=device))
print(CKPT_LIST[model_index])

if MULTIGPU:
    # multi-GPU setting
    model = nn.DataParallel(model, device_ids = GPU_ID)
    model = model.to(device)
else:
    # single-GPU setting
    model = model.to(device)

model = model.eval()