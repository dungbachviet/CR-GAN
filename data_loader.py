# Xi Peng, Feb 2017
# Yu Tian, Apr 2017
# Manage some of operations: manage how to get dataset (transfrom, resize,...) 
import os, sys
import numpy as np
from PIL import Image
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import pdb

dd = pdb.set_trace
# 9 viewpoints (they all were rotated evenly)('200','190'... are only symbols of the author)
views = ['200', '190', '041', '050', '051', '140', '130', '080', '090']

pi = 3.1416 # 180 degree
d_60 = pi / 3
d_15 = pi / 12
d_range = pi / 36 # 5 degree

d_45 = d_60 - d_15
d_30 = d_45 - d_15

def read_img(img_path):
    # img_path: /home/yt219/data/multi_PIE_crop_128/192/192_01_02_140_07_crop_128.png
    img = Image.open(img_path).convert('RGB') # read image and return in format of a digital array
    img = img.resize((128,128), Image.ANTIALIAS)
    return img

# Get a pair of (viewpoint, image) from MultiPIE dataset
def get_multiPIE_img(img_path):
    # img_path: /home/yt219/data/multi_PIE_crop_128/192/192_01_02_140_07_crop_128.png
    tmp = random.randint(0, 8) # get a random digit from (>= 0 and <=8) representing for index 1 of 9 viewpoint indices
    view2 = tmp
    view = views[tmp]
    token = img_path.split('/')
    name = token[-1]
        
    token = name.split('_')
    ID = token[0]
    status = token[2]
    bright = token[4]
    # Simply create an image of other viewpoint (view2)
    img2_path = '/home/yt219/data/multi_PIE_crop_128/' + ID + '/' + ID + '_01_' + status + '_' + view + '_' + bright + '_crop_128.png'
    img2 = read_img( img2_path ) # read the new-view image and return its digital array
    img2 = img2.resize((128,128), Image.ANTIALIAS)
    return view2, img2 # return image of new viewpoint

# Get a pair of (view, image) from 300w_LP dataset
def get_300w_LP_img(img_path):
    # img_path = '/home/yt219/data/crop_0822/AFW_resize/AFW_1051618982_1_0_128.jpg'
    # txt_path: /home/yt219/data/300w_LP_size_128/AFW_resize/AFW_1051618982_1_0_128_pose_shape_expression_128.txt 
    right = img_path.find('_128.jpg')
    for i in range(right-1, 0, -1):
        if img_path[i] == '_':
            left = i
            break
    
    # This dataset needs a .txt file to identify the view of one image
    # This step to randomly choose one image belong to 1 in 9 specified viewpoints
    view2 = -1
    while(view2 < 0):
        tmp = random.randint(0, 17)
        new_txt = img_path[:left+1] + str(tmp) + '_128_pose_shape_expression_128.txt'
        new_txt = new_txt.replace("crop_0907", "300w_LP_size_128")
        
        if os.path.isfile(new_txt):
            param = np.loadtxt(new_txt)
            yaw = param[1]
            if yaw < -d_60 or yaw > d_60:
                view2 = -1
            elif yaw >= -d_60 and yaw < -d_60+d_range:
                view2 = 0
            elif yaw >= -d_45-d_range and yaw < -d_45+d_range:
                view2 = 1
            elif yaw >= -d_30-d_range and yaw < -d_30+d_range:
                view2 = 2
            elif yaw >= -d_15-d_range and yaw < -d_15+d_range:
                view2 = 3
            elif yaw >= -d_range and yaw < d_range:
                view2 = 4
            elif yaw >= d_15-d_range and yaw < d_15+d_range:
                view2 = 5
            elif yaw >= d_30-d_range and yaw < d_30+d_range:
                view2 = 6
            elif yaw >= d_45-d_range and yaw < d_45+d_range:
                view2 = 7
            elif yaw >= d_60-d_range and yaw <= d_60:
                view2 = 8
    
    # When find out the suited image 
    # Read that image and return its digital arrays that has been resized into (128,128) dimension
    new_img = img_path[:left+1] + str(tmp) + '_128.jpg'
    img2 = read_img( new_img )
    img2 = img2.resize((128,128), Image.ANTIALIAS)
    return view2, img2

# Class for managing the process of get images from dataset
# This class inherits from module of data.Dataset for integrating some useful transfromations and other utilities
class ImageList(data.Dataset):
    # We can put in some necessary parameters here (to get out useful information from dataset)
    def __init__( self, list_file, transform=None, is_train=True, 
                  img_shape=[128,128] ):
        img_list = [line.rstrip('\n') for line in open(list_file)] # list_file contains image paths (once per each line)
        print('total %d images' % len(img_list))

        self.img_list = img_list
        self.transform = transform
        self.is_train = is_train
        self.img_shape = img_shape
        self.transform_img = transforms.Compose([self.transform]) # contain a chain of image transformation operations 

    
    # Automatically get the two random pairs of (view, image) (same identity at 2 viewpoints) for one training instance
    # index: the system automatically create the random value for index in range of 0-->len(dataset)-1
    def __getitem__(self, index):
        # img_name: /home/yt219/data/multi_PIE_crop_128/192/192_01_02_140_07_crop_128.png
        img1_path = self.img_list[index]
        token = img1_path.split(' ') # split to get 2 informations: image path and its viewpoint
        img1_fpath = token[0] # image path
        view1 = int(token[1]) # viewpoint
        
        img1 = read_img(img1_fpath) # read the image and return its digital array
        
        # Check whether to get images from MultiPIE or 300wLP dataset
        if img1_fpath.find('multi_PIE') > -1:
            view2, img2 = get_multiPIE_img(img1_fpath)
        else:
            view2, img2 = get_300w_LP_img(img1_fpath)
        
        # Check whether or not using image transformation
        if self.transform_img is not None:
            img1 = self.transform_img(img1) # [0,1], c x h x w
            img2 = self.transform_img(img2)

        return view1, view2, img1, img2 # Return 2 images of 2 viewpoints of the one same identity
    
    # Size of dataset
    def __len__(self):
        return len(self.img_list)
