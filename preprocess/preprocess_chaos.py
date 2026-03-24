import os
import shutil

import numpy  as np 
import cv2
import os
from glob import glob
from PIL import Image
from collections import defaultdict

raw_pixel_counts = defaultdict(int)      # 处理前统计
processed_pixel_counts = defaultdict(int)  # 处理后统计

def crop_to_square(img):
    # 选择图像的较小维度，并裁剪为正方形
    h, w = img.shape[:2]
    size = min(h, w)
    top = (h - size) // 2
    left = (w - size) // 2
    cropped_img = img[top:top + size, left:left + size]
    return cropped_img

def resize_with_aspect_ratio_and_pad(img, target_size, interpolation):
    # 首先裁剪为正方形
    img = crop_to_square(img)
    
    # 然后缩放到目标大小
    resized_img = cv2.resize(img, (target_size, target_size), interpolation=interpolation)
    return resized_img

client_name = ['0']
client_num = len(client_name)
data_path = '/mnt/diskC/zhw/my_nnunet/nnUNet_raw/Dataset003_kva_syn'
# data_path = '/mnt/diskC/zhw/my_nnunet/fundus_syn_all'
target_path = '/mnt/diskB/zhw/polyp_syn_1024_new'
if not os.path.exists(target_path):
    os.makedirs(target_path)
client_data_list = []
slice_num =[]

for client_idx in range(client_num):
    print('{}/imagesTr/*'.format(data_path))
    client_data_list.append(glob('{}/imagesTr/*'.format(data_path)))
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
        img_data = np.array(Image.open(filename))
        labelname = filename.replace('images','labels').replace('_0000','')
        if os.path.exists(labelname):
            label_data = np.array(Image.open(labelname))
            # 统计处理前分布
            unique_vals_raw, counts_raw = np.unique(label_data, return_counts=True)
            for val, count in zip(unique_vals_raw, counts_raw):
                raw_pixel_counts[val] += count
            image_file_name = os.path.basename(filename)
            rgb_image = resize_with_aspect_ratio_and_pad(img_data, 1024, cv2.INTER_LINEAR)

            # rgb_image = np.stack([rgb_image] * 3, axis=-1) # only for chaos
            image_new_name = image_file_name+'.npy'
            # print(rgb_image.shape)
            save_path = os.path.join(dir_name,image_new_name)
            # print(save_path)
            rgb_image=rgb_image[:,:,:3]
            np.save(save_path,rgb_image)
            label = label_data.astype(np.uint8)
            # label = cv2.resize(label, (256,256), interpolation=cv2.INTER_NEAREST)[:,:,:1]
            label = resize_with_aspect_ratio_and_pad(label, 256, cv2.INTER_NEAREST)
            
            # label[label < 150] = 0        
            # label[label == 255] = 1
            # 统计处理后分布
            unique_vals_post, counts_post = np.unique(label, return_counts=True)
            for val, count in zip(unique_vals_post, counts_post):
                processed_pixel_counts[val] += count
            save_path = os.path.join(label_dir_name,image_new_name)
            # print(save_path)
            
            np.save(save_path,label)
            num = num + 1

print(num)
print("\n=== 原始标签像素分布 ===")
for val in sorted(raw_pixel_counts.keys()):
    print(f"像素值: {val}, 总数: {raw_pixel_counts[val]}")

print("\n=== 处理后标签像素分布 ===")
for val in sorted(processed_pixel_counts.keys()):
    print(f"像素值: {val}, 总数: {processed_pixel_counts[val]}")
