import os
import shutil

import numpy  as np 
import cv2
import os
from glob import glob
from PIL import Image
client_name = ['fundus234', 'fundus134', 'fundus124', 'fundus123']
client_num = len(client_name)
data_path = '/home/zhanghanwen/fundus_1024'
target_path = '/home/zhanghanwen/fundus_syn_disk'
if not os.path.exists(target_path):
    os.makedirs(target_path)
client_data_list = []
slice_num =[]
for client_idx in range(client_num):
    print('{}/{}/gen-disk/*'.format(data_path,client_name[client_idx]))
    client_data_list.append(glob('{}/{}/gen-disk/*'.format(data_path,client_name[client_idx])))
    print (len(client_data_list[client_idx]))
    slice_num.append(len(client_data_list[client_idx]))
for client_idx in range(client_num):
    dir_name = '{}/{}/data_npy/'.format(target_path,client_name[client_idx])
    label_dir_name = '{}/{}/label_npy/'.format(target_path,client_name[client_idx])
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if not os.path.exists(label_dir_name):
        os.makedirs(label_dir_name)
    for fid, filename in enumerate(client_data_list[client_idx]):
        img_data = np.array(Image.open(filename))
        img_data = np.array(Image.open(filename).convert('RGB'))
        labelname = filename.replace('gen','masks256')
        label_data = np.array(Image.open(labelname).convert("L"))
        image_file_name = os.path.basename(filename)
        rgb_image=cv2.resize(img_data, (1024,1024), interpolation=cv2.INTER_LINEAR)
        image_new_name = image_file_name+'.npy'
        # print(rgb_image.shape)
        save_path = os.path.join(dir_name,image_new_name)
            # print(save_path)
        np.save(save_path,rgb_image)
        label = label_data
        label[label != 0] = 1
        label = np.expand_dims(label.astype(np.uint8),axis=-1)   
        print(label.shape)
        save_path = os.path.join(label_dir_name,image_new_name)
        # print(save_path)
        np.save(save_path,label)