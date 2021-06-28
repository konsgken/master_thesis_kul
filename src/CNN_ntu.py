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
import pathlib

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MEAN = np.load('../data/nyu/nyu_mean.npy')
STD = np.load('../data/nyu/nyu_std.npy')
model_dir = '../model/NYU.pth'
dataset_dir = '../data/NTU/'
center_dir = '../data/NTU/Image/'
save_dir = './result/NTU'

depth_thres = 150
batch_size = 64


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
        label = label.long()
        label = label - 1
        return data, label

    def __len__(self):
        return len(self.depth_images)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 176 * 176, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64 * 176 * 176)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


if __name__ == '__main__':
    cropWidth = 176
    cropHeight = 176
    epochs = 35
    learning_rate = 0.03

    image_dataset = ntu_dataloader(dataset_dir, center_dir, depth_thres)
    train_samples = int(len(image_dataset) * 0.8)
    test_samples = len(image_dataset) - train_samples
    train_set, test_set = torch.utils.data.random_split(image_dataset, (train_samples, test_samples))
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                   shuffle=False, num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                  shuffle=False, num_workers=2)
    net = Net()
    # net = net.cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                        filename=os.path.join(save_dir, 'train.log'), level=logging.INFO)
    logging.info('======================================================')

    for epoch in range(epochs):
        net = net.train()
        total = 0
        training_running_loss = 0.0
        training_running_corrects = 0
        training_loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=True)
        test_running_loss = 0.0
        test_running_corrects = 0

        for i, (img, label) in training_loop:
            # torch.cuda.synchronize()
            # img, label = img.cuda(), label.cuda()
            optimizer.zero_grad()
            preds = net(img)
            _, predicted = torch.max(preds.data, 1)
            total += label.size(0)
            loss = criterion(preds, label)
            loss.backward()
            optimizer.step()

            training_running_loss += loss.item() * img.size(0)
            training_running_corrects += (predicted == label).sum().item()

            training_loop.set_description(f"Epoch [{epoch}/{epochs}]")
            training_loop.set_postfix(batch_loss=loss.item(), training_accuracy=100 * training_running_corrects / total)
        epoch_train_loss = training_running_loss / train_samples
        epoch_train_acc = training_running_corrects / train_samples

        net = net.eval()
        for i, (img, label) in enumerate(test_dataloader):
            with torch.no_grad():
                preds = net(img)
                _, predicted = torch.max(preds.data, 1)
                loss = criterion(preds, label)
                test_running_loss += loss.item() * img.size(0)
                test_running_corrects += (predicted == label).sum().item()

        epoch_test_loss = test_running_loss / test_samples
        epoch_test_acc = test_running_corrects / test_samples

        print("[INFO] ", "Training Loss: ", epoch_train_loss, " Training Accuracy: ", epoch_train_acc, " Test Loss: ",
              epoch_test_loss, " Test Accuracy: ", epoch_test_acc)
        logging.info('Epoch#%d: total training loss=%.4f, total training accuracy=%.4f, total test loss=%.4f, '
                     'test accuracy=%.4f '
                     % (epoch, epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc))
    torch.save(net.state_dict(), '../src_train/result/NTU/NTU_hand_gesture_weights' + '.pth')