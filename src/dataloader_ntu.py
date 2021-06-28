import glob
import pandas as pd
import cv2
import torch
import torch.utils.data
import numpy as np
import scipy.io as scio
import os
import matplotlib.pyplot as plt
import pathlib
import torchvision
from torch.autograd import Variable
import model as model
import anchor as anchor
from tqdm import tqdm

MEAN = np.load('../data/nyu/nyu_mean.npy')
STD = np.load('../data/nyu/nyu_std.npy')
model_dir = '../model/NYU.pth'
dataset_dir = '../data/NTU/'
center_dir = '../data/NTU/Image/'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
depth_thres = 150
batch_size = 10


def normalize(x):
    return 255 * (np.array((x - np.min(x)) / (np.max(x) - np.min(x)))).astype('uint8')


class ntu_dataloader(torch.utils.data.Dataset):

    def __init__(self, ImgDir, center_dir, depth_thres):
        self.ImgDir = ImgDir
        self.mean = MEAN
        self.std = STD
        self.h = 176
        self.w = 176
        self.depth_thres = depth_thres
        self.rgb_images = glob.glob(ImgDir + 'Image/**/**/*.jpg')
        self.depth_images = glob.glob(ImgDir + 'Depth/**/**/*.txt')

    def __getitem__(self, index):
        # depth size: (480, 640), depth.max(): 2001, depth.min():733
        gesture_id = os.path.basename(os.path.dirname(self.depth_images[index]))
        person_id = os.path.basename(os.path.dirname(os.path.dirname(self.depth_images[index])))
        depth = np.loadtxt(self.depth_images[index], delimiter=',')
        label = int(gesture_id.split('G')[-1])
        center = pd.read_csv(self.ImgDir + 'Image/' + person_id + '/' + gesture_id + '/' + 'labels_' + person_id + '_' + gesture_id + '.csv', header=None)
        x_center, y_center = \
            center[center.iloc[:, 3].str.match(pathlib.Path(os.path.basename(self.depth_images[index])).stem + '.jpg')].iloc[:, [1, 2]].values[0]
        x_center = x_center - 25
        data = depth[int(y_center - (self.h / 2)): int(y_center + (self.h / 2)),
               int(x_center - (self.w / 2)): int(x_center + (self.w / 2))]
        data = cv2.resize(data, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        # data, label = dataPreprocess(index, depth, self.keypointsUVD, self.center, self.mean, self.std, \
        #                             self.lefttop_pixel, self.rightbottom_pixel, self.xy_thres, self.depth_thres)
        center_cropped_img = data[int(data.shape[0] / 2), int(data.shape[1] / 2)]
        data = np.asarray(data, dtype='float32')
        data[np.where(data >= center_cropped_img + depth_thres)] = center_cropped_img
        data[np.where(data <= center_cropped_img - depth_thres)] = center_cropped_img
        data = data - center_cropped_img
        data = (data - MEAN) / STD
        data = data[np.newaxis, ...]
        label = np.asarray(label, dtype='float32')
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        return data, label

    def __len__(self):
        return len(self.depth_images)


if __name__ == '__main__':
    test_image_datasets = ntu_dataloader(dataset_dir, center_dir, depth_thres)
    test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=batch_size,
                                                   shuffle=False, num_workers=8)
    plt.rcParams["figure.figsize"] = (12, 8)
    iterator = iter(test_dataloaders)
    for ii in range(20):
        data, labels = next(iterator)
        # if ii <= 20:
        #     continue
        data = data.numpy()
        labels = labels.numpy()
        fig, axs = plt.subplots(2, 5)
        img = data[0, :, :, :].squeeze()
        axs[0, 0].imshow(data[0, :, :, :].squeeze(), cmap='gray', vmin=data[0, :, :, :].squeeze().min(),
                         vmax=data[0, :, :, :].squeeze().max())
        axs[0, 0].set_title('Gesture ' + str(int(labels[0])))
        axs[0, 1].imshow(data[1, :, :, :].squeeze(), cmap='gray', vmin=data[1, :, :, :].squeeze().min(),
                         vmax=data[0, :, :, :].squeeze().max())
        axs[0, 1].set_title('Gesture ' + str(int(labels[1])))
        axs[0, 2].imshow(data[2, :, :, :].squeeze(), cmap='gray', vmin=data[2, :, :, :].squeeze().min(),
                         vmax=data[0, :, :, :].squeeze().max())
        axs[0, 2].set_title('Gesture ' + str(int(labels[2])))
        axs[0, 3].imshow(data[3, :, :, :].squeeze(), cmap='gray', vmin=data[3, :, :, :].squeeze().min(),
                         vmax=data[0, :, :, :].squeeze().max())
        axs[0, 3].set_title('Gesture ' + str(int(labels[3])))
        axs[0, 4].imshow(data[4, :, :, :].squeeze(), cmap='gray', vmin=data[4, :, :, :].squeeze().min(),
                         vmax=data[0, :, :, :].squeeze().max())
        axs[0, 4].set_title('Gesture ' + str(int(labels[4])))
        axs[1, 0].imshow(data[5, :, :, :].squeeze(), cmap='gray', vmin=data[5, :, :, :].squeeze().min(),
                         vmax=data[0, :, :, :].squeeze().max())
        axs[1, 0].set_title('Gesture ' + str(int(labels[5])))
        axs[1, 1].imshow(data[6, :, :, :].squeeze(), cmap='gray', vmin=data[6, :, :, :].squeeze().min(),
                         vmax=data[0, :, :, :].squeeze().max())
        axs[1, 1].set_title('Gesture ' + str(int(labels[6])))
        axs[1, 2].imshow(data[7, :, :, :].squeeze(), cmap='gray', vmin=data[7, :, :, :].squeeze().min(),
                         vmax=data[0, :, :, :].squeeze().max())
        axs[1, 2].set_title('Gesture ' + str(int(labels[7])))
        axs[1, 3].imshow(data[8, :, :, :].squeeze(), cmap='gray', vmin=data[8, :, :, :].squeeze().min(),
                         vmax=data[0, :, :, :].squeeze().max())
        axs[1, 3].set_title('Gesture ' + str(int(labels[8])))
        axs[1, 4].imshow(data[9, :, :, :].squeeze(), cmap='gray', vmin=data[9, :, :, :].squeeze().min(),
                         vmax=data[0, :, :, :].squeeze().max())
        axs[1, 4].set_title('Gesture ' + str(int(labels[9])))
