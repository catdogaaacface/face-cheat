from PIL import Image
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import pickle
import os
from tqdm import tqdm
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152

from util import extract_features,utils


IMG_DIR="/home/zhangao/projects/face-attack/dataset/securityAI_round1_dirimages"
ID_DIR="/home/zhangao/projects/face-attack/dataset/securityAI_round1_dirimages"
ADV_DIR='/home/zhangao/projects/face-attack/dataset/pgd_Linf_eps12_iter20_ens4mask'
MASK_DIR='/home/zhangao/projects/face-attack/dataset/securityAI_round1_mask'

if not os.path.exists(ADV_DIR):
    os.mkdir(ADV_DIR)
MULTIGPU=False

CKPT_LIST =['backbone_resnet50.pth',
            'backbone_ir50_ms1m.pth',
            'backbone_ir50_asia.pth',
            'backbone_ir152.pth'
            ]
CKPT_LIST = [os.path.join('/home/zhangao/model_zoo/face_model_zoo',ckpt) for ckpt in CKPT_LIST]

model_list = [ResNet_50([112,112]),
              IR_50([112,112]),
              IR_50([112,112]),
              IR_152([112,112])
              ]
weights = []
LOAD_EMBEDDINGS=None
#LOAD_EMBEDDINGS="id_embeddings_rgb.pkl"
PIN_MEMORY=False
NUM_WORKERS=0
BATCH_SIZE=8
EMBEDDING_SIZE = 512
SAVE = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

for model,ckpt in zip(model_list,CKPT_LIST):
    model.to(device)
    model.load_state_dict(torch.load(ckpt,map_location=device))
    model.eval()

# Loading Test Data (Un-normalized)
transform_test = transforms.Compose([transforms.ToTensor(),])
dataset = datasets.ImageFolder(IMG_DIR,transform=transform_test)

loader = torch.utils.data.DataLoader(
    dataset, batch_size = BATCH_SIZE,  pin_memory = PIN_MEMORY,
    num_workers = NUM_WORKERS
)

# Attacking Images batch-wise
def attack(model_list,  img, label, id_embeddings_list, eps, attack_type, iters,device,target_or_not,target_index):
    #model_list=model_list[1:]
    adv = img.detach()
    adv.requires_grad = True

    if target_or_not:
        flag = -1
    else:
        flag = 1
    
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
        loss=0
        for model,id_embeddings in zip(model_list,id_embeddings_list):
            features = extract_features.l2_normlize(model(utils.normalize(adv.clone())))
            loss = loss + utils.distance_loss(id_embeddings,features,target_index,device)
            #losses.append(utils.distance_loss(id_embeddings,features,target_index,device))
        #loss = torch.sum(torch.cat(losses))
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
            adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
            adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
        adv.data.clamp_(0.0, 1.0)
        adv.grad.data.zero_()
    return adv.detach()

# Attacking Images batch-wise
def attack_in_mask(model_list,  img, label,mask, id_embeddings_list, eps, attack_type, iters,device,target_or_not,target_index):
    #model_list=model_list[1:]
    adv = img.detach()
    adv.requires_grad = True

    if target_or_not:
        flag = -1
    else:
        flag = 1
    
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
        loss=0
        for model,id_embeddings in zip(model_list,id_embeddings_list):
            features = extract_features.l2_normlize(model(utils.normalize(adv.clone())))
            loss = loss + utils.distance_loss(id_embeddings,features,target_index,device)
            #losses.append(utils.distance_loss(id_embeddings,features,target_index,device))
        #loss = torch.sum(torch.cat(losses))
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
            adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
            adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
            adv.data = adv.data * mask + img.data * (1-mask)
        adv.data.clamp_(0.0, 1.0)
        adv.grad.data.zero_()
    return adv.detach()

# if  not LOAD_EMBEDDINGS==None:
#     fr = open(LOAD_EMBEDDINGS,'rb')
#     id_embeddings,id_name_to_index,id_index_to_name = pickle.load(fr)

# else:
carray,id_name_to_index,id_index_to_name,maskarray = utils.read_identities(ID_DIR,MASK_DIR)
maskarray = torch.from_numpy(maskarray).float().to(device)
id_embeddings_list = []
for model in model_list:
    id_embeddings_list.append(extract_features.generate_id_embeddings(MULTIGPU,device,EMBEDDING_SIZE,BATCH_SIZE,model,carray))
    # fw = open('id_embeddings_dee.pkl','wb')
    # pickle.dump([id_embeddings,id_name_to_index,id_index_to_name],fw)
    # fw.close()

target_index,untarget_index = utils.generate_target_index(dataset.class_to_idx,dataset.samples,id_name_to_index) 


adv_acc = [0]*len(model_list)
clean_acc = [0]*len(model_list)
adv_success = 0
eps =12.0/255.0 # Epsilon for Adversarial Attack
id_embeddings_t_lists=[]
for id_embeddings in id_embeddings_list:
    id_embeddings_t_lists.append(torch.from_numpy(id_embeddings.T).float().to(device))


for i, (img, label) in enumerate(tqdm(loader)):
    batch_target_index = target_index[i*BATCH_SIZE:i*BATCH_SIZE+len(label.numpy())]
    batch_untarget_index = untarget_index[i*BATCH_SIZE:i*BATCH_SIZE+len(label.numpy())]
    img = img.to(device)
    batch_mask = maskarray[batch_untarget_index]
    #print(img.numpy().shape)
    # for k,(model,id_embeddings_t) in enumerate(zip(model_list,id_embeddings_t_lists)):
    #     features =extract_features.l2_normlize( model(utils.normalize(img.clone().detach())))
    #     _pred = features.mm(id_embeddings_t).argmax(dim=-1).detach().cpu().numpy()
    #     pred= np.zeros(label.numpy().shape[0],dtype=np.int)
    #     for j,index in enumerate(_pred):
    #         if not id_index_to_name[index] in dataset.class_to_idx:
    #             pred[j]=-1
    #         else:
    #             pred[j] = dataset.class_to_idx[id_index_to_name[index]]
    #     clean_acc[k] += np.sum(pred==label.numpy())
    
    adv= attack(model_list, img, label,id_embeddings_list, eps=eps, attack_type= 'pgd', iters= 20,device=device, target_or_not=False, target_index=batch_untarget_index)
    adv= attack_in_mask(model_list, img, label,batch_mask, id_embeddings_list, eps=eps, attack_type= 'pgd', iters= 20,device=device, target_or_not=False, target_index=batch_untarget_index)
    
    if SAVE:
        for j,arr in enumerate(adv.clone().detach().cpu().numpy()): 
            dataset.samples[i*BATCH_SIZE+j][0].split('/')[-1]
            img = Image.fromarray((255*arr.transpose(1,2,0)).astype(np.uint8))
            # namedir = os.path.join('adv_ens_pgd8',dataset.classes[label.numpy()[j]])
            # if not os.path.exists(namedir):
            #     os.mkdir(namedir)
            img_name = os.path.basename(dataset.samples[j+i*BATCH_SIZE][0]).split('.')[0]
            #print(os.path.join(ADV_DIR,"%s"%(dataset.samples[i*BATCH_SIZE+j][0].split('/')[-1])))           
            img.save(os.path.join(ADV_DIR,"%s"%(dataset.samples[i*BATCH_SIZE+j][0].split('/')[-1])))
            #img.save("_adv_img/%s-%s.jpg"%(dataset.classes[label.numpy()[j]],id_index_to_name[batch_target_index[j]]))
            #img.save("_adv_img/%s-%s.jpg"%(dataset.classes[label.numpy()[j]],id_index_to_name[batch_target_index[j]]))
#     for k,(model,id_embeddings_t) in enumerate(zip(model_list,id_embeddings_t_lists)):
#         features_adv = extract_features.l2_normlize( model(utils.normalize(adv.clone().detach())))
#         _pred_adv = features_adv.mm(id_embeddings_t).argmax(dim=-1).detach().cpu().numpy()
#         pred_adv= np.zeros(label.numpy().shape[0],dtype=np.int)
#         for j,index in enumerate(_pred_adv):
#             if not id_index_to_name[index] in dataset.class_to_idx:
#                 pred_adv[j]=-1
#             else:
#                 pred_adv[j] = dataset.class_to_idx[id_index_to_name[index]]
#         adv_acc[k] += np.sum(pred_adv==label.numpy())
#         #adv_success += np.sum(_pred_adv==np.array(batch_target_index))

# adv_acc = [val/len(dataset) for val in adv_acc]
# clean_acc =[val/len(dataset) for val in clean_acc]
# adv_success = adv_success/len(dataset)
# for i in range(len(model_list)):
#     print('Clean accuracy:{0:.3%}\t Adversarial accuracy:{1:.3%}\t Attack success rate:{2:.3%}'.format(clean_acc[i] , adv_acc[i], adv_success))



