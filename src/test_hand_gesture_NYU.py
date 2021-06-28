import cv2
import torch
import torch.utils.data
import numpy as np
import scipy.io as scio
import os
from PIL import Image
from torch.autograd import Variable
import model as model
import anchor as anchor
from tqdm import tqdm
import glob
import pandas as pd
import cv2
import torch
import torch.utils.data
import numpy as np
import scipy.io as scio
import os
import matplotlib.pyplot as plt
import torchvision
from torch.autograd import Variable
import model as model
import anchor as anchor
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
from tqdm import tqdm
import logging
import time
import datetime
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

fx = 588.03
fy = -587.07
u0 = 320
v0 = 240

# DataHyperParms
keypointsNumber = 14
cropWidth = 176
cropHeight = 176
batch_size = 20
xy_thres = 110
depth_thres = 150

save_dir = '../result/NYU'

try:
    os.makedirs(save_dir)
except OSError:
    pass

testingImageDir = '../data/nyu/nyu_hand_dataset_v2/Preprossed/test_nyu/'  # mat images
center_file = '../data/nyu/nyu_center_test.mat'
MEAN = np.load('../data/nyu/nyu_mean.npy')
STD = np.load('../data/nyu/nyu_std.npy')
model_dir = '../model/NYU.pth'
keypoint_file = '../data/nyu/nyu_keypointsUVD_test.mat'
result_file = 'result_NYU.txt'


def pixel2world(x, fx, fy, ux, uy):
    x[:, :, 0] = (x[:, :, 0] - ux) * x[:, :, 2] / fx
    x[:, :, 1] = (x[:, :, 1] - uy) * x[:, :, 2] / fy
    return x


def world2pixel(x, fx, fy, ux, uy):
    x[:, :, 0] = x[:, :, 0] * fx / x[:, :, 2] + ux
    x[:, :, 1] = x[:, :, 1] * fy / x[:, :, 2] + uy
    return x


# Keypoints of the test set. Size (8252,14,3)
keypointsUVD_test = scio.loadmat(keypoint_file)['keypoints3D'].astype(np.float32)
# Centre UVD coordinate of the keypoint (8252,1,3)
center_test = scio.loadmat(center_file)['centre_pixel'].astype(np.float32)
# Convert UVD coordinate of the centre to XYZ coordinates (8252,1,3)
centre_test_world = pixel2world(center_test.copy(), fx, fy, u0, v0)

centerlefttop_test = centre_test_world.copy()
centerlefttop_test[:, 0, 0] = centerlefttop_test[:, 0, 0] - xy_thres
centerlefttop_test[:, 0, 1] = centerlefttop_test[:, 0, 1] + xy_thres

centerrightbottom_test = centre_test_world.copy()
centerrightbottom_test[:, 0, 0] = centerrightbottom_test[:, 0, 0] + xy_thres
centerrightbottom_test[:, 0, 1] = centerrightbottom_test[:, 0, 1] - xy_thres

test_lefttop_pixel = world2pixel(centerlefttop_test, fx, fy, u0, v0)
test_rightbottom_pixel = world2pixel(centerrightbottom_test, fx, fy, u0, v0)


def dataPreprocess(index, img, keypointsUVD, center, mean, std, lefttop_pixel, rightbottom_pixel, xy_thres=90,
                   depth_thres=75):
    imageOutputs = np.ones((cropHeight, cropWidth, 1), dtype='float32')
    labelOutputs = np.ones((keypointsNumber, 3), dtype='float32')

    new_Xmin = max(lefttop_pixel[index, 0, 0], 0)
    new_Ymin = max(lefttop_pixel[index, 0, 1], 0)
    new_Xmax = min(rightbottom_pixel[index, 0, 0], img.shape[1] - 1)
    new_Ymax = min(rightbottom_pixel[index, 0, 1], img.shape[0] - 1)

    # crop the original depth maps according to center points, which give us a hand-centered sub-image :(169,170)
    imCrop = img[int(new_Ymin):int(new_Ymax), int(new_Xmin):int(new_Xmax)].copy()
    # Resized crooped image : (176,176)
    imgResize = cv2.resize(imCrop, (cropWidth, cropHeight), interpolation=cv2.INTER_NEAREST)

    imgResize = np.asarray(imgResize, dtype='float32')  # H*W*C

    imgResize[np.where(imgResize >= center[index][0][2] + depth_thres)] = center[index][0][2]
    imgResize[np.where(imgResize <= center[index][0][2] - depth_thres)] = center[index][0][2]
    imgResize = (imgResize - center[index][0][2])

    imgResize = (imgResize - mean) / std

    ## label
    label_xy = np.ones((keypointsNumber, 2), dtype='float32')
    label_xy[:, 0] = (keypointsUVD[index, :, 0].copy() - new_Xmin) * cropWidth / (new_Xmax - new_Xmin)  # x
    label_xy[:, 1] = (keypointsUVD[index, :, 1].copy() - new_Ymin) * cropHeight / (new_Ymax - new_Ymin)  # y

    imageOutputs[:, :, 0] = imgResize

    labelOutputs[:, 1] = label_xy[:, 0]
    labelOutputs[:, 0] = label_xy[:, 1]

    labelOutputs[:, 2] = (keypointsUVD[index, :, 2] - center[index][0][2])  # Z

    imageOutputs = np.asarray(imageOutputs)
    imageNCHWOut = imageOutputs.transpose(2, 0, 1)  # [H, W, C] --->>>  [C, H, W]
    imageNCHWOut = np.asarray(imageNCHWOut)
    labelOutputs = np.asarray(labelOutputs)

    data, label = torch.from_numpy(imageNCHWOut), torch.from_numpy(labelOutputs)

    return data, label


######################   Pytorch dataloader   #################
class my_dataloader(torch.utils.data.Dataset):

    def __init__(self, ImgDir, center, lefttop_pixel, rightbottom_pixel, keypointsUVD):
        self.ImgDir = ImgDir
        self.mean = MEAN
        self.std = STD
        self.center = center
        self.lefttop_pixel = lefttop_pixel
        self.rightbottom_pixel = rightbottom_pixel
        self.keypointsUVD = keypointsUVD
        self.xy_thres = xy_thres
        self.depth_thres = depth_thres

    def __getitem__(self, index):
        # depth size: (480, 640), depth.max(): 2001, depth.min():733
        depth = scio.loadmat(self.ImgDir + str(index + 1) + '.mat')['depth']

        data, label = dataPreprocess(index, depth, self.keypointsUVD, self.center, self.mean, self.std, \
                                     self.lefttop_pixel, self.rightbottom_pixel, self.xy_thres, self.depth_thres)

        return data, label

    def __len__(self):
        return len(self.center)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(26896, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 41 * 41)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    predicted_list = []
    test_image_datasets = my_dataloader(testingImageDir, center_test, test_lefttop_pixel, test_rightbottom_pixel,
                                                keypointsUVD_test)
    test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=batch_size,
                                               shuffle=False, num_workers=8)
    net = Net()
    net.load_state_dict(torch.load( '../src_train/result/NTU/NTU_hand_gesture_weights' + '.pth'))
    net.eval()
    for i, (img, _) in tqdm(enumerate(test_dataloaders)):
        if i >= 100:
            break
        with torch.no_grad():

            preds = net(img)
            _, predicted = torch.max(preds.data, 1)
            predicted_list.append((predicted.numpy()))

