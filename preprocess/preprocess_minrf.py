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
data_path = '/mnt/diskB/zhw/Dataset104_Nuclei-minRF/imagesTr'
# data_path = '/mnt/diskC/zhw/my_nnunet/fundus_syn_all'
target_path = '/mnt/diskB/zhw/nuclei_1024_rf'
if not os.path.exists(target_path):
    os.makedirs(target_path)
client_data_list = []
slice_num =[]
for client_idx in range(client_num):
    print('{}/*'.format(data_path))
    client_data_list.append(glob('{}/*'.format(data_path)))
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
        labelname = filename.replace('imagesTr','labelsTr_real').replace('_0000.png','.png')
        if os.path.exists(labelname):
            label_data = np.array(Image.open(labelname))
            image_file_name = os.path.basename(filename)
            rgb_image=cv2.resize(img_data, (1024,1024), interpolation=cv2.INTER_LINEAR)
            # rgb_image = np.stack([rgb_image] * 3, axis=-1) # only for chaos
            image_new_name = image_file_name+'.npy'
            # print(rgb_image.shape)
            save_path = os.path.join(dir_name,image_new_name)
            # print(save_path)
            rgb_image=rgb_image[:,:,:3]
            np.save(save_path,rgb_image)
            label = label_data.astype(np.uint8)
            # label = cv2.resize(label, (256,256), interpolation=cv2.INTER_NEAREST)[:,:,:1]
            label = cv2.resize(label, (256,256), interpolation=cv2.INTER_NEAREST)
            
            label[label < 128] = 0        
            label[label >= 128] = 1
            unique_vals, counts = np.unique(label, return_counts=True)
            for val, count in zip(unique_vals, counts):
                global_pixel_counts[val] += count
            save_path = os.path.join(label_dir_name,image_new_name)
            # print(save_path)
            
            np.save(save_path,label)

print("\n=== 全体数据标签像素分布 ===")
for val in sorted(global_pixel_counts.keys()):
    print(f"像素值: {val}, 总数: {global_pixel_counts[val]}")