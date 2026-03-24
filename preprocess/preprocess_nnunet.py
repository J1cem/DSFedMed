import os
import shutil

import numpy  as np 
import cv2
import os
from glob import glob
from PIL import Image
from collections import defaultdict
global_pixel_counts = defaultdict(int)
client_name = ['0']
client_num = len(client_name)
data_path = '/mnt/diskC/zhw/my_nnunet/nnUNet_preprocessed/Dataset001_syn_all/nnUNetPlans_2d'
# data_path = '/mnt/diskC/zhw/my_nnunet/fundus_syn_all'
target_path = '/mnt/diskB/zhw/polyp_syn_1024_new'
if not os.path.exists(target_path):
    os.makedirs(target_path)
client_data_list = []
slice_num =[]
for client_idx in range(client_num):
    print('{}/*'.format(data_path))
    client_data_list.append(glob('{}/*mask.npy'.format(data_path)))
    print (len(client_data_list[client_idx]))
    slice_num.append(len(client_data_list[client_idx]))
num = 0
for client_idx in range(client_num):
    dir_name = '{}/{}/data_npy/'.format(target_path,client_name[client_idx])
    label_dir_name = '{}/{}/label_npy/'.format(target_path,client_name[client_idx])
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if not os.path.exists(label_dir_name):
        os.makedirs(label_dir_name)
    for fid, filename in enumerate(client_data_list[client_idx]):
        labelname = filename.replace('mask.npy','mask_seg.npy')
        if os.path.exists(labelname):
            name = os.path.basename(filename)
            save_path = os.path.join(dir_name,name)
            shutil.copy(filename, save_path)
            label_save_path = os.path.join(label_dir_name,name)
            shutil.copy(labelname, label_save_path)
            num = num + 1
print(num)
