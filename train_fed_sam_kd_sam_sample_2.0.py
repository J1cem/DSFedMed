import os
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0,1,2,3,4,5,6,7]))
# 自动设置 TMPDIR（如果目录不存在就创建）
tmpdir = "/mnt/diskC/.../tmpdir"
os.makedirs(tmpdir, exist_ok=True)
os.environ["TMPDIR"] = tmpdir

import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import copy
import shutil
import argparse
import logging
import time
import random
import numpy as np
import collections
from collections import OrderedDict
from glob import glob
import cv2
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.utils import make_grid
from pytorch_metric_learning import losses

from torch.utils.data import Subset, DataLoader
from networks.unet2d import Unet2D
from utils.losses import dice_loss, softmax_kl_loss, segmentation_kl_loss, softmax_mse_loss, symmetric_mse_loss, kl_loss, segmentation_bce_loss, segmentation_kl_loss_2
from utils.util import _eval_dice, _eval_haus, _connectivity_region_analysis, parse_fn_haus
from dataloaders.federated_dataloader import Dataset, ToTensor, ProstateDataset
from tinysam import sam_model_registry
# from models.sam import SamPredictor, sam_model_registry
from models.sam.utils.transforms import ResizeLongestSide
from sam_utils import *
# stu_net = sam_model_registry['vit_t'](args,checkpoint='/mnt/diskD/hanwen/tinysam.pth').to(device)
import copy
import cfg
args = cfg.parse_args()
snapshot_path = "/mnt/diskC/.../fed-sam-kd-output/weight-0.2/" + args.exp + "/"
batch_size = args.batch_size * len(args.gpu.split(','))
meta_step_size = args.meta_step_size
clip_value = args.clip_value
base_lr = args.base_lr
# client_num = args.client_num
max_epoch = args.max_epoch
display_freq = args.display_freq
# client names and paths for different federated learning datasets
client_name = ['1', '6', '18', '21']
data_path = '/mnt/diskB/.../FeTS2022_FedDG_1024'
if args.data=='Prostate_od':
    client_name =  ['BIDMC', 'HK', 'I2CVB', 'ISBI', 'ISBI_1.5', 'UCL']
    data_path = '/mnt/diskB/.../Prostate_processed_1024'
    syn_data_path = '/mnt/diskB/.../prostate_cn_1024'
elif args.data=='Nuclei_od':
    client_name = ['MoNuSAC2020','TNBC','MoNuSAC2018']
    data_path = '/mnt/diskB/.../Nuclei_1024'
    syn_data_path = '/mnt/diskB/.../nuclei_cn_1024_3'
elif args.data=='fundus_cup':
    client_name = ['fundus1', 'fundus2', 'fundus3', 'fundus4']
    syn_data_path = '/home/.../fundus_syn_cup'
    data_path = '/home/.../fundus_1024_256_cup'
elif args.data=='fundus_disk':
    client_name = ['fundus1', 'fundus2', 'fundus3', 'fundus4']
    syn_data_path = '/home/.../fundus_syn_disk'
    data_path = '/home/.../fundus_1024_256_disk'
elif args.data=='Chaos':
    client_name = ['0', '1']
    data_path = '/mnt/diskB/.../chaos_1024'
    syn_data_path = '/mnt/diskB/.../chaos_syn_1024'
elif args.data=='ISIC':
    client_name = ['0', '1', '2']
    data_path = '/mnt/diskB/.../isic_1024'
    syn_data_path = '/mnt/diskB/.../isic_syn_1024'
client_num = len(client_name)
client_data_list = []
client_val_data_list = []
slice_num =[]
for client_idx in range(client_num):
    print('{}/{}/data_npy/*'.format(data_path,client_name[client_idx]))
    client_data_list.append(glob('{}/{}/data_npy/*'.format(data_path,client_name[client_idx])))
    client_val_data_list.append(glob('{}/{}/val_data_npy/*'.format(data_path,client_name[client_idx])))
    print (len(client_data_list[client_idx]),len(client_val_data_list[client_idx]))
    slice_num.append(len(client_data_list[client_idx]))
# print(client_val_data_list)
slice_num = np.array(slice_num)
#volume_size = [384, 384, 3]
unseen_site_idx = args.unseen_site
client_data_list[unseen_site_idx].extend(client_val_data_list[unseen_site_idx])
print('unseen site data length:',len(client_data_list[unseen_site_idx]))
source_site_idx = [i for i in range(client_num)]
source_site_idx.remove(unseen_site_idx)
client_weight = slice_num[source_site_idx] / np.sum(slice_num[source_site_idx])
client_weight = np.round(client_weight, decimals=2)
client_weight[-1] = 1 - np.sum(client_weight[:-1])
# client_weight = np.insert(client_weight, unseen_site_idx, 0)
print(client_weight)
# client_weight= np.full((client_num,), 1/client_num)
# client_weight[-1] = 1 - np.sum(client_weight[:2])
if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
print(torch.__version__)
print(torch.cuda.is_available())

def evaluate_sample_losses(model, dataloader, criterion, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            volume_batch_raw_np = images[:, :3, ...]
            labels = batch['label'].to(device)
            pt = batch['pt'].to(device)
            point_labels = torch.ones(images.size(0))
            if point_labels[0] != -1:
                point_coords = pt
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                pt = (coords_torch, labels_torch)
            se, de = model.prompt_encoder(
                        points=pt,
                        boxes=None,
                        masks=None
                    )
            volume_batch_raw, label_batch = \
                    volume_batch_raw_np.cuda(GPUdevice), labels.cuda(GPUdevice)
            # image,mask = image.cuda(GPUdevice), mask.cuda(GPUdevice)
            image_encoded= model.image_encoder(volume_batch_raw)
            pred, _, _ = model.mask_decoder(
                image_embeddings=image_encoded,
                image_pe=model.prompt_encoder.get_dense_pe(), #  1x(embed_dim)x(embedding_h)x(embedding_w)
                sparse_prompt_embeddings=se,
                dense_prompt_embeddings=de, 
                multimask_output=False,
            )
            loss = criterion(pred, label_batch)
            loss_per_sample = loss.view(loss.size(0), -1).mean(dim=1)
            losses.extend(loss_per_sample.detach().cpu().numpy())
    return np.array(losses)


def select_hard_samples(losses, ratio=0.1):
    k = int(len(losses) * ratio)
    hard_indices = np.argsort(-losses)[:k]
    return hard_indices.tolist()

def get_hard_sample_dataloader(dataset, hard_indices, batch_size):
    hard_subset = Subset(dataset, hard_indices)
    dataloader = DataLoader(hard_subset, batch_size=batch_size, shuffle=True)
    return dataloader

# The global model is obtained by weighted aggregation of data volume
def update_global_model(net_clients, client_weight):
    for param in zip(*list(net_clients[i].parameters() for i in range(client_num))):
        # Only the trained parameters need to be aggregated
        if param[0].requires_grad is False:
            continue
        new_para = Variable(torch.Tensor(np.zeros(param[0].shape)), requires_grad=False).cuda(GPUdevice) 
        for i in range(client_num):
            new_para.data.add_(client_weight[i], param[i].data)

        for i in range(client_num):
            param[i].data.mul_(0).add_(new_para.data)

def distribute_global_model(global_model, net_clients):
    for param_global, *params_clients in zip(global_model.parameters(), *(client.parameters() for client in net_clients)):
        for param_client in params_clients:
            param_client.data.copy_(param_global.data)

def update_global_model_all(net_clients, client_weight):
    """
    Aggregates the state_dict (including both parameters and buffers) of all clients 
    and updates all client models with the aggregated global model.
    """
    # 获取所有客户端的 state_dict（包括参数和 buffers）
    state_dicts = [net.state_dict() for net in net_clients]
    
    # 初始化全局 state_dict
    global_state = {}

    # 聚合所有参数和 buffers
    for key in state_dicts[0].keys():
        global_state[key] = sum(
            client_weight[i] * state_dicts[i][key] for i in range(len(net_clients))
        )

    # 将聚合后的 state_dict 加载回每个客户端模型
    for net in net_clients:
        net.load_state_dict(global_state)


def val_unet(site_index, test_net):
    val_data_list = client_val_data_list[site_index]
    if site_index == -1:
        val_data_list = [item for sublist in client_val_data_list for item in sublist]
    dice_array = []
    eiou_array = []
    dice_array_cup = []
    eiou_array_cup = []
    test_net.eval()
    for fid, filename in enumerate(val_data_list):
        # The image and its mask are read in one at a time
        data = np.load(filename)/ 255.0
        mask_data = np.load(filename.replace("data", "label"))
        if data.ndim == 2 or data.shape[-1] == 1:  # Grayscale image (H, W) or (H, W, 1)
            data = np.repeat(np.expand_dims(data, axis=-1), 3, axis=-1)
        image = np.expand_dims(data[..., :3].transpose(2, 0, 1), axis=0)
        if mask_data.ndim == 2:  # (H, W)
            mask_data = np.expand_dims(mask_data, axis=-1) 
        mask = np.expand_dims(mask_data.transpose(2, 0, 1), axis=0)
        image = torch.from_numpy(image).float().cuda(GPUdevice)
        mask = torch.from_numpy(mask).cuda(GPUdevice)
        # The embedding of prompt is removed in decoder, so pt is set to 0
        pt = np.expand_dims(np.array([0,0]), axis=0)
        point_labels = torch.ones(image.size(0))
        if point_labels[0] != -1:
            point_coords = pt
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
            pt = (coords_torch, labels_torch)
        '''
        se, de = test_net.prompt_encoder(
            points=pt,
            boxes=None,
            masks=None
        )
        '''
        # image,mask = image.cuda(GPUdevice), mask.cuda(GPUdevice)
        # image_encoded= test_net.image_encoder(image)
        pred, _, _ = test_net(image)
        pred = pred.cuda(GPUdevice)
        # dice and iou are calculated by different thresholds
        threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
        temp = eval_seg(pred, mask, threshold)
        # fundus needs to segment out oc and od, so there are two pred result
        if(pred.shape[1]==2):
            iou_d, iou_c, disc_dice, cup_dice= temp
            dice_array.append(disc_dice)
            eiou_array.append(iou_d)
            dice_array_cup.append(cup_dice)
            eiou_array_cup.append(iou_c)
        else:
            eiou, edice = temp
            dice_array.append(edice)
            eiou_array.append(eiou)
    # Calculate the total result
    if args.num_classes==2:
        dice_array = np.array(dice_array)
        eiou_array = np.array(eiou_array)
        dice_array_cup = np.array(dice_array_cup)
        eiou_array_cup = np.array(eiou_array_cup)
        dice_avg = np.mean(dice_array, axis=0).tolist()
        eiou_avg = np.mean(eiou_array, axis=0).tolist()
        dice_avg_cup = np.mean(dice_array_cup, axis=0).tolist()
        eiou_avg_cup = np.mean(eiou_array_cup, axis=0).tolist()
        logging.info("validate data from client %d Disc Dice %.4f, Disc IOU %.4f, Cup Dice %.4f, Cup IOU %.4f" % (site_index, dice_avg, eiou_avg, dice_avg_cup, eiou_avg_cup))
        return dice_avg,eiou_avg, dice_avg_cup, eiou_avg_cup
    else:
        dice_array = np.array(dice_array)
        eiou_array = np.array(eiou_array)
        # print (dice_array.shape)
        dice_avg = np.mean(dice_array, axis=0).tolist()
        eiou_avg = np.mean(eiou_array, axis=0).tolist()
        if site_index != -1:
            logging.info("validate data from client %d OD dice_avg %.4f, Eiou %.4f" % (site_index, dice_avg, eiou_avg))
        return dice_avg, eiou_avg

def val(site_index, test_net):
    val_data_list = client_val_data_list[site_index]
    if site_index == -1:
        val_data_list = [item for sublist in client_val_data_list for item in sublist]
    dice_array = []
    eiou_array = []
    dice_array_cup = []
    eiou_array_cup = []
    test_net.eval()
    for fid, filename in enumerate(val_data_list):
        # The image and its mask are read in one at a time
        data = np.load(filename)/ 255.0
        mask_data = np.load(filename.replace("data", "label"))
        if data.ndim == 2 or data.shape[-1] == 1:  # Grayscale image (H, W) or (H, W, 1)
            data = np.repeat(np.expand_dims(data, axis=-1), 3, axis=-1)
        image = np.expand_dims(data[..., :3].transpose(2, 0, 1), axis=0)
        if mask_data.ndim == 2:  # (H, W)
            mask_data = np.expand_dims(mask_data, axis=-1) 
        mask = np.expand_dims(mask_data.transpose(2, 0, 1), axis=0)
        image = torch.from_numpy(image).float().cuda(GPUdevice)
        mask = torch.from_numpy(mask).cuda(GPUdevice)
        # The embedding of prompt is removed in decoder, so pt is set to 0
        pt = np.expand_dims(np.array([0,0]), axis=0)
        point_labels = torch.ones(image.size(0))
        if point_labels[0] != -1:
            point_coords = pt
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
            pt = (coords_torch, labels_torch)
        se, de = test_net.prompt_encoder(
            points=pt,
            boxes=None,
            masks=None
        )
        # image,mask = image.cuda(GPUdevice), mask.cuda(GPUdevice)
        image_encoded= test_net.image_encoder(image)
        pred, _, _ = test_net.mask_decoder(
            image_embeddings=image_encoded,
            image_pe=test_net.prompt_encoder.get_dense_pe(), #  1x(embed_dim)x(embedding_h)x(embedding_w)
            sparse_prompt_embeddings=se,
            dense_prompt_embeddings=de, 
            multimask_output=False,
        )
        # dice and iou are calculated by different thresholds
        threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
        temp = eval_seg(pred, mask, threshold)
        # fundus needs to segment out oc and od, so there are two pred result
        if(pred.shape[1]==2):
            iou_d, iou_c, disc_dice, cup_dice= temp
            dice_array.append(disc_dice)
            eiou_array.append(iou_d)
            dice_array_cup.append(cup_dice)
            eiou_array_cup.append(iou_c)
        else:
            eiou, edice = temp
            dice_array.append(edice)
            eiou_array.append(eiou)
    # Calculate the total result
    if args.num_classes==2:
        dice_array = np.array(dice_array)
        eiou_array = np.array(eiou_array)
        dice_array_cup = np.array(dice_array_cup)
        eiou_array_cup = np.array(eiou_array_cup)
        dice_avg = np.mean(dice_array, axis=0).tolist()
        eiou_avg = np.mean(eiou_array, axis=0).tolist()
        dice_avg_cup = np.mean(dice_array_cup, axis=0).tolist()
        eiou_avg_cup = np.mean(eiou_array_cup, axis=0).tolist()
        logging.info("validate data from client %d Disc Dice %.4f, Disc IOU %.4f, Cup Dice %.4f, Cup IOU %.4f" % (site_index, dice_avg, eiou_avg, dice_avg_cup, eiou_avg_cup))
        return dice_avg,eiou_avg, dice_avg_cup, eiou_avg_cup
    else:
        dice_array = np.array(dice_array)
        eiou_array = np.array(eiou_array)
        # print (dice_array.shape)
        dice_avg = np.mean(dice_array, axis=0).tolist()
        eiou_avg = np.mean(eiou_array, axis=0).tolist()
        if site_index != -1:
            logging.info("validate data from client %d OD dice_avg %.4f, Eiou %.4f" % (site_index, dice_avg, eiou_avg))
        return dice_avg, eiou_avg
# Validation is performed on the validation set of each client
def validation_unet(test_net):
    dice_avg, eiou_avg, dice_avg_cup, eiou_avg_cup =[], [], [], []
    for i in source_site_idx:
        if args.num_classes==2:
            dice_avg_one, eiou_avg_one, dice_avg_cup_one, eiou_avg_cup_one=val_unet(i,test_net)
            dice_avg.append(dice_avg_one)
            eiou_avg.append(eiou_avg_one)
            dice_avg_cup.append(dice_avg_cup_one)
            eiou_avg_cup.append(eiou_avg_cup_one)
        else:
            dice_avg_one, eiou_avg_one=val_unet(i,test_net)
            dice_avg.append(dice_avg_one)
            eiou_avg.append(eiou_avg_one)

    dice_avg, eiou_avg = val(-1,test_net)
    if args.num_classes==2:
        dice_avg_cup = np.mean(np.array(dice_avg_cup), axis=0)
        eiou_avg_cup = np.mean(np.array(eiou_avg_cup), axis=0)
        logging.info("Averagy validation: Disc Dice %.4f, Disc IOU %.4f, Cup Dice %.4f, Cup IOU %.4f" % (dice_avg, eiou_avg, dice_avg_cup, eiou_avg_cup))
        return dice_avg, eiou_avg, dice_avg_cup, eiou_avg_cup
    else:
        logging.info("Averagy validation: OD dice_avg %.4f, Eiou %.4f" % (dice_avg, eiou_avg))
        return dice_avg, eiou_avg

def validation(test_net):
    dice_avg, eiou_avg, dice_avg_cup, eiou_avg_cup =[], [], [], []
    for i in source_site_idx:
        if args.num_classes==2:
            dice_avg_one, eiou_avg_one, dice_avg_cup_one, eiou_avg_cup_one=val(i,test_net)
            dice_avg.append(dice_avg_one)
            eiou_avg.append(eiou_avg_one)
            dice_avg_cup.append(dice_avg_cup_one)
            eiou_avg_cup.append(eiou_avg_cup_one)
        else:
            dice_avg_one, eiou_avg_one=val(i,test_net)
            dice_avg.append(dice_avg_one)
            eiou_avg.append(eiou_avg_one)
    
    dice_avg, eiou_avg = val(-1,test_net)
    if args.num_classes==2:
        dice_avg_cup = np.mean(np.array(dice_avg_cup), axis=0)
        eiou_avg_cup = np.mean(np.array(eiou_avg_cup), axis=0)
        logging.info("Averagy validation: Disc Dice %.4f, Disc IOU %.4f, Cup Dice %.4f, Cup IOU %.4f" % (dice_avg, eiou_avg, dice_avg_cup, eiou_avg_cup))
        return dice_avg, eiou_avg, dice_avg_cup, eiou_avg_cup
    else:
        logging.info("Averagy validation: OD dice_avg %.4f, Eiou %.4f" % (dice_avg, eiou_avg))
        return dice_avg, eiou_avg
def save_image(mask_data,name='pred'):
    mask_show= mask_data[0,0,:,:].copy()
    mask_show=mask_show*255
    image_save = Image.fromarray(mask_show).convert("RGB")
    image_save.save("../output/output-fundus-disc-{}.jpg".format(name))
    mask_show= mask_data[0,1,:,:].copy()
    mask_show=mask_show*255
    image_save = Image.fromarray(mask_show).convert("RGB")
    image_save.save("../output/output-fundus-cup-{}.jpg".format(name))
    
# Test the data on unseensite using the aggregated global model
def test_unet(site_index, test_net):

    test_data_list = client_data_list[site_index]

    dice_array = []
    eiou_array = []
    dice_array_cup = []
    eiou_array_cup = []
    # print(test_data_list)
    test_net.eval()
    # print('test',site_index,len(test_data_list))
    for fid, filename in enumerate(test_data_list):
        # The image and its mask are read in one at a time
        data = np.load(filename)/ 255.0
        mask_data = np.load(filename.replace("data", "label"))
        if data.ndim == 2 or data.shape[-1] == 1:  # Grayscale image (H, W) or (H, W, 1)
            data = np.repeat(np.expand_dims(data, axis=-1), 3, axis=-1)
        image = np.expand_dims(data[..., :3].transpose(2, 0, 1), axis=0)
        if mask_data.ndim == 2:  # (H, W)
            mask_data = np.expand_dims(mask_data, axis=-1) 
        mask = np.expand_dims(mask_data.transpose(2, 0, 1), axis=0)
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask)
        # The embedding of prompt is removed in decoder, so pt is set to 0
        pt = np.expand_dims(np.array([0,0]), axis=0)
        point_labels = torch.ones(image.size(0))
        if point_labels[0] != -1:
            point_coords = pt
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
            pt = (coords_torch, labels_torch)
        '''
        se, de = test_net.prompt_encoder(
            points=pt,
            boxes=None,
            masks=None
        )
        '''
        image,mask = image.cuda(GPUdevice), mask.cuda(GPUdevice)
        # image_encoded= test_net.image_encoder(image)
        pred, _, _ = test_net(image)
        pred = pred.cuda(GPUdevice)
        # dice and iou are calculated by different thresholds
        threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
        temp = eval_seg(pred, mask, threshold)
        # fundus needs to segment out oc and od, so there are two pred result
        if(pred.shape[1]==2):
            iou_d, iou_c, disc_dice, cup_dice= temp
            dice_array.append(disc_dice)
            eiou_array.append(iou_d)
            dice_array_cup.append(cup_dice)
            eiou_array_cup.append(iou_c)
        else:
            eiou, edice = temp
            dice_array.append(edice)
            eiou_array.append(eiou)
    # Calculate the total result
    if args.num_classes==2:
        dice_array = np.array(dice_array)
        eiou_array = np.array(eiou_array)
        dice_array_cup = np.array(dice_array_cup)
        eiou_array_cup = np.array(eiou_array_cup)
        dice_avg = np.mean(dice_array, axis=0).tolist()
        eiou_avg = np.mean(eiou_array, axis=0).tolist()
        dice_avg_cup = np.mean(dice_array_cup, axis=0).tolist()
        eiou_avg_cup = np.mean(eiou_array_cup, axis=0).tolist()
        logging.info("Test Disc Dice %.4f, Disc IOU %.4f, Cup Dice %.4f, Cup IOU %.4f" % (dice_avg, eiou_avg, dice_avg_cup, eiou_avg_cup))
        return dice_avg,eiou_avg, dice_avg_cup, eiou_avg_cup
    else:
        dice_array = np.array(dice_array)
        eiou_array = np.array(eiou_array)
        # print (dice_array.shape)
        dice_avg = np.mean(dice_array, axis=0).tolist()
        eiou_avg = np.mean(eiou_array, axis=0).tolist()
        logging.info("Test OD dice_avg %.4f, Eiou %.4f" % (dice_avg, eiou_avg))
        return dice_avg, eiou_avg

def test(site_index, test_net):

    test_data_list = client_data_list[site_index]

    dice_array = []
    eiou_array = []
    dice_array_cup = []
    eiou_array_cup = []
    # print(test_data_list)
    test_net.eval()
    # print('test',site_index,len(test_data_list))
    for fid, filename in enumerate(test_data_list):
        # The image and its mask are read in one at a time
        data = np.load(filename)/ 255.0
        mask_data = np.load(filename.replace("data", "label"))
        if data.ndim == 2 or data.shape[-1] == 1:  # Grayscale image (H, W) or (H, W, 1)
            data = np.repeat(np.expand_dims(data, axis=-1), 3, axis=-1)
        image = np.expand_dims(data[..., :3].transpose(2, 0, 1), axis=0)
        if mask_data.ndim == 2:  # (H, W)
            mask_data = np.expand_dims(mask_data, axis=-1) 
        mask = np.expand_dims(mask_data.transpose(2, 0, 1), axis=0)
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask)
        # The embedding of prompt is removed in decoder, so pt is set to 0
        pt = np.expand_dims(np.array([0,0]), axis=0)
        point_labels = torch.ones(image.size(0))
        if point_labels[0] != -1:
            point_coords = pt
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
            pt = (coords_torch, labels_torch)
        se, de = test_net.prompt_encoder(
            points=pt,
            boxes=None,
            masks=None
        )
        image,mask = image.cuda(GPUdevice), mask.cuda(GPUdevice)
        image_encoded= test_net.image_encoder(image)
        pred, _, _ = test_net.mask_decoder(
            image_embeddings=image_encoded,
            image_pe=test_net.prompt_encoder.get_dense_pe(), #  1x(embed_dim)x(embedding_h)x(embedding_w)
            sparse_prompt_embeddings=se,
            dense_prompt_embeddings=de, 
            multimask_output=False,
        )
        # dice and iou are calculated by different thresholds
        threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
        temp = eval_seg(pred, mask, threshold)
        # fundus needs to segment out oc and od, so there are two pred result
        if(pred.shape[1]==2):
            iou_d, iou_c, disc_dice, cup_dice= temp
            dice_array.append(disc_dice)
            eiou_array.append(iou_d)
            dice_array_cup.append(cup_dice)
            eiou_array_cup.append(iou_c)
        else:
            eiou, edice = temp
            dice_array.append(edice)
            eiou_array.append(eiou)
    # Calculate the total result
    if args.num_classes==2:
        dice_array = np.array(dice_array)
        eiou_array = np.array(eiou_array)
        dice_array_cup = np.array(dice_array_cup)
        eiou_array_cup = np.array(eiou_array_cup)
        dice_avg = np.mean(dice_array, axis=0).tolist()
        eiou_avg = np.mean(eiou_array, axis=0).tolist()
        dice_avg_cup = np.mean(dice_array_cup, axis=0).tolist()
        eiou_avg_cup = np.mean(eiou_array_cup, axis=0).tolist()
        logging.info("******Test Disc Dice %.4f, Disc IOU %.4f, Cup Dice %.4f, Cup IOU %.4f" % (dice_avg, eiou_avg, dice_avg_cup, eiou_avg_cup))
        return dice_avg,eiou_avg, dice_avg_cup, eiou_avg_cup
    else:
        dice_array = np.array(dice_array)
        eiou_array = np.array(eiou_array)
        # print (dice_array.shape)
        dice_avg = np.mean(dice_array, axis=0).tolist()
        eiou_avg = np.mean(eiou_array, axis=0).tolist()
        logging.info("******Test OD dice_avg %.4f, Eiou %.4f" % (dice_avg, eiou_avg))
        return dice_avg, eiou_avg

def copy_outer_net(fast_weights,net_current):
    # Deep copy the net current model
    net_copy = copy.deepcopy(net_current)
    # Assign the fast weights to the net copy model
    for name, param in net_copy.named_parameters():
        if name in fast_weights:
            param.data = fast_weights[name]
    return net_copy

if __name__ == "__main__":
    # 日志和设备设置
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    GPUdevice = torch.device('cuda', int(args.gpu))
    logging.info(str(args))

    # 构造合成数据集 DataLoader
    dataset_list = []
    for client_idx in range(client_num):
        if "prostate" in args.data:
            dataset = ProstateDataset(client_idx=client_idx, data_path=syn_data_path,freq_site_idx=client_idx,
                                split='train', transform = transforms.Compose([
                                ToTensor(),
                                ]),client_name=client_name)
        else:
            dataset = Dataset(client_idx=client_idx, data_path=syn_data_path,freq_site_idx=client_idx, split='train', transform = transforms.Compose([ToTensor(),]),client_name=client_name)
        if "od" in args.data:
            if client_idx != args.unseen_site:
                dataset_list.append(dataset)
        else:
            dataset_list.append(dataset)
    syn_dataset = ConcatDataset(dataset_list)
    global_dataloader = DataLoader(syn_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                   pin_memory=True, worker_init_fn=lambda id: random.seed(args.seed + id))
    print(f"Total dataset size: {len(syn_dataset)}")
    print(f"Number of batches: {len(global_dataloader)}")

    # 初始化教师模型（主模型）与学生模型
    teacher_net = sam_model_registry['vit_b'](args, checkpoint=args.tea_ckpt).to(GPUdevice)
    student_net = sam_model_registry['vit_t'](args, checkpoint=args.stu_ckpt).to(GPUdevice)
    if "fundus" in args.data:
        args.num_classes = 2
        baseline_net = sam_model_registry['vit_b'](args, checkpoint=args.baseline_ckpt).to(GPUdevice)
        args.num_classes = 1
    else:
        baseline_net = sam_model_registry['vit_b'](args, checkpoint=args.baseline_ckpt).to(GPUdevice)
    
    teacher_net.train()
    student_net.train()
    # 冻结 teacher 模型的 image encoder
    for param in teacher_net.image_encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(student_net.parameters(), lr=args.base_lr, betas=(0.9, 0.999))
    tea_optimizer = torch.optim.Adam(teacher_net.parameters(), lr=args.base_lr, betas=(0.9, 0.999))
    # 蒸馏损失函数设置
    lossfunc = torch.nn.BCEWithLogitsLoss(pos_weight=torch.ones([1]).cuda(GPUdevice) * 2)
    lossfunc_pixelwise = torch.nn.BCEWithLogitsLoss(pos_weight=torch.ones([1]).cuda(GPUdevice) * 2, reduction='none')
    loss_weights = 0.9
    start_kd = args.kd_epoch
    feature_loss = args.feature

    print("=== Pre-Trained FedTinySAM Model ===")
    dice_avg, eiou_avg = test(unseen_site_idx, student_net)
    print("Test OD dice is: {}, IOU is {}".format(dice_avg,eiou_avg))
    print("=== Pre-Trained SAM Model ===")
    dice_avg, eiou_avg = test(unseen_site_idx, teacher_net)
    print("Test OD dice is: {}, IOU is {}".format(dice_avg,eiou_avg))
    print("=== Pre-Trained Baeline FedSAM Model ===")
    # dice_avg, eiou_avg = test(unseen_site_idx, baseline_net)
    # print("Test OD dice is: {}, IOU is {}".format(dice_avg,eiou_avg))
    print("=== Evaluating Tiny SAM Model ===")
    validation(student_net)
    print("=== Evaluating Global SAM Model ===")
    validation(teacher_net)
    print("=== Evaluating Baeline FedSAM Model ===")
    # validation(baseline_net)

    best_val_dice = 0.0
    best_model_path = os.path.join(snapshot_path + '/model', 'best_model.pth')
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    
    # 蒸馏训练开始
    k_ratio = 0.5

    for epoch_num in range(args.start_epoch, max_epoch):
        for i_batch, sampled_batch in enumerate(global_dataloader):
            volume_batch, label_batch, pt = sampled_batch['image'], sampled_batch['label'], sampled_batch['pt']
            volume_batch_raw_np = volume_batch[:, :3, ...]
            point_labels = torch.ones(volume_batch.size(0))

            if point_labels[0] != -1:
                point_coords = pt
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=GPUdevice)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=GPUdevice)
                pt = (coords_torch[None, :, :], labels_torch[None, :])

            with torch.no_grad():
                se_teacher, de_teacher = teacher_net.prompt_encoder(points=pt, boxes=None, masks=None)
                se_student, de_student = student_net.prompt_encoder(points=pt, boxes=None, masks=None)

            volume_batch_raw = volume_batch_raw_np.cuda(GPUdevice)
            label_batch = label_batch.cuda(GPUdevice)

            feat_teacher = teacher_net.image_encoder(volume_batch_raw)
            out_teacher, _, _ = teacher_net.mask_decoder(
                image_embeddings=feat_teacher,
                image_pe=teacher_net.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se_teacher,
                dense_prompt_embeddings=de_teacher,
                multimask_output=False,
            )

            feat_student = student_net.image_encoder(volume_batch_raw)
            out_student, _, _ = student_net.mask_decoder(
                image_embeddings=feat_student,
                image_pe=student_net.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=se_student,
                dense_prompt_embeddings=de_student,
                multimask_output=False,
            )

            # Supervision losses
            loss_gt_teacher = lossfunc(out_teacher, label_batch)
            loss_gt_student = lossfunc(out_student, label_batch)
            loss_feat = symmetric_mse_loss(feat_student, feat_teacher)

            # KL losses
            loss_kl_1, loss_kl_1_each = kl_loss(out_teacher, out_student, temperature=1.0, flag='sample')  # T -> S
            loss_kl_2, loss_kl_2_each = kl_loss(out_student, out_teacher, temperature=1.0, flag='sample')  # S -> T

            # Batch size
            B = volume_batch.shape[0]

            with torch.no_grad():
                # 每个样本的 supervision loss
                loss_teacher_each = lossfunc_pixelwise(out_teacher, label_batch).view(B, -1).mean(dim=1)
                loss_student_each = lossfunc_pixelwise(out_student, label_batch).view(B, -1).mean(dim=1)

                # ========== 分位数筛选可靠样本 ==========
                reliable_ratio = 0.7  # 保留 GT loss 最小的 70% 样本

                teacher_thresh = torch.quantile(loss_teacher_each, reliable_ratio)
                student_thresh = torch.quantile(loss_student_each, reliable_ratio)

                reliable_mask = (loss_teacher_each <= teacher_thresh) & (loss_student_each <= student_thresh)

                if reliable_mask.sum() == 0:
                    reliable_mask[:] = True  # fallback防止选空

                # 计算 KL 差异
                loss_kl_1_map = loss_kl_1_each.view(B, -1).mean(dim=1)
                loss_kl_2_map = loss_kl_2_each.view(B, -1).mean(dim=1)
                kl_sum = loss_kl_1_map + loss_kl_2_map

                # 把不可靠样本排除掉
                kl_sum[~reliable_mask] = -1e9  # 极值 确保不会被选中

                # 选择 KL 最大的一部分可靠样本
                topk = min(max(int(k_ratio * B), 1), B)
                _, idx = torch.topk(kl_sum, topk, largest=True)
                idx = idx.long()


            # 蒸馏 loss 只在 selected sample 上计算
            loss_kl_1_selected = loss_kl_1_each[idx].mean()
            loss_kl_2_selected = loss_kl_2_each[idx].mean()

            # Combine loss
            if epoch_num < start_kd:
                total_loss = loss_gt_teacher
            else:
                total_loss = (
                    loss_weights * loss_gt_teacher +
                    loss_weights * loss_gt_student +
                    (1 - loss_weights) * loss_kl_1_selected +
                    (1 - loss_weights) * loss_kl_2_selected +
                    (loss_feat if feature_loss else 0)
                )

            optimizer.zero_grad()
            tea_optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            tea_optimizer.step()

            if i_batch % display_freq == 0:
                logging.info(f"[Epoch {epoch_num}] Batch {i_batch}: "
                             f"GT Loss T: {loss_gt_teacher.item():.4f}, "
                             f"GT Loss S: {loss_gt_student.item():.4f}, "
                             f"KL Loss: {(loss_kl_1_selected + loss_kl_2_selected).item():.4f}, "
                             f"Feature Loss: {loss_feat.item():.4f}, "
                             f"Total: {total_loss.item():.4f}")
            
        print("=== Evaluating Tiny SAM Model ===")
        dice, iou = validation(student_net)
        print("=== Evaluating Global SAM Model ===")
        validation(teacher_net)
        # === Save best model based on validation dice ===
        if dice > best_val_dice:
            best_val_dice = dice
            # k_ratio = min(k_ratio + 0.1, 1.0)
            torch.save(student_net.state_dict(), best_model_path)
            print(f"==> Saved best model at epoch {epoch_num} with dice {dice:.4f}")
            dice_avg, eiou_avg = test(unseen_site_idx, student_net)
            logging.info(("Best Student Test OD dice is: {}, IOU is {}".format(dice_avg,eiou_avg)))

        if epoch_num % 5==0 or epoch_num==max_epoch-1:
            ## evaluation test unseen
            with open(os.path.join(snapshot_path, 'evaluation_result.txt'), 'a') as f:
                dice_list = []
                haus_list = []
                print("epoch {} testing , site {}".format(epoch_num, unseen_site_idx), file=f)
                
                dice_avg, eiou_avg= test(unseen_site_idx, student_net)
                print(("Student Test OD dice is: {}, IOU is {}".format(dice_avg,eiou_avg)),file=f)
                dice_avg, eiou_avg = test(unseen_site_idx, teacher_net)
                print(("Teacher Test OD dice is: {}, IOU is {}".format(dice_avg,eiou_avg)),file=f)
            '''
            if epoch_num % 25==0:    
                save_mode_path = os.path.join(snapshot_path + '/model', 'epoch_' + str(epoch_num) + '.pth')
                os.makedirs(os.path.dirname(save_mode_path), exist_ok=True)
                torch.save(student_net.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
            '''
        
    print("\n=== Final Testing with Best Validation Model ===")
    student_net.load_state_dict(torch.load(best_model_path))
    student_net.eval()

    dice_avg, eiou_avg = test(unseen_site_idx, student_net)
    logging.info(("Best Student Model Test OD dice is: {}, IOU is {}".format(dice_avg, eiou_avg)))
