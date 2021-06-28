import cv2
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import scipy.io as scio
import os
from PIL import Image
from torch.autograd import Variable
import resnet18_group_normalization_model as model
import anchor as anchor
from tqdm import tqdm
import random_erasing
import logging
import time
import datetime
import random
import pathlib
import pandas as pd
import glob
import torch.nn as nn
from torchvision import transforms
from PIL import Image
print('RESNET GROUP 18 Group Normalization = 16')
# from torchviz import make_dot
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

fx = 588.03
fy = -587.07
u0 = 320
v0 = 240

# DataHyperParms
TrainImgFrames = 72757
TestImgFrames = 8252
keypointsNumber = 14
cropWidth = 176
cropHeight = 176
batch_size = 64
learning_rate = 0.00035
Weight_Decay = 1e-4
nepoch = 35
RegLossFactor = 3
spatialFactor = 0.5
RandCropShift = 5
RandshiftDepth = 1
RandRotate = 180
RandScale = (1.0, 0.5)
xy_thres = 110
depth_thres = 150

randomseed = 12345
random.seed(randomseed)
np.random.seed(randomseed)
torch.manual_seed(randomseed)

save_dir = '../result/NYU_batch_64_12345'

try:
    os.makedirs(save_dir)
except OSError:
    pass
dataset_dir = '../data/NTU/'
center_dir = '../data/NTU/Image/'
save_dir = './result/NTU'

trainingImageDir = '../data/nyu/nyu_hand_dataset_v2/Preprossed/train_nyu/'
testingImageDir = '../data/nyu/nyu_hand_dataset_v2/Preprossed/test_nyu/'  # mat images
test_center_file = '../data/nyu/nyu_center_test.mat'
test_keypoint_file = '../data/nyu/nyu_keypointsUVD_test.mat'
train_center_file = '../data/nyu/nyu_center_train.mat'
train_keypoint_file = '../data/nyu/nyu_keypointsUVD_train.mat'
MEAN = np.load('../data/nyu/nyu_mean.npy')
STD = np.load('../data/nyu/nyu_std.npy')
model_dir = '../model/NYU.pth'
result_file = 'result_NYU.txt'


def pixel2world(x, fx, fy, ux, uy):
    x[:, :, 0] = (x[:, :, 0] - ux) * x[:, :, 2] / fx
    x[:, :, 1] = (x[:, :, 1] - uy) * x[:, :, 2] / fy
    return x


def world2pixel(x, fx, fy, ux, uy):
    x[:, :, 0] = x[:, :, 0] * fx / x[:, :, 2] + ux
    x[:, :, 1] = x[:, :, 1] * fy / x[:, :, 2] + uy
    return x


joint_id_to_name = {
    0: 'pinky tip',
    1: 'pinky mid',
    2: 'ring tip',
    3: 'ring mid',
    4: 'middle tip',
    5: 'middle mid',
    6: 'index tip',
    7: 'index mid',
    8: 'thumb tip',
    9: 'thumb mid',
    10: 'thumb root',
    11: 'wrist back',
    12: 'wrist',
    13: 'palm',
}

## loading GT keypoints and center points
keypointsUVD_test = scio.loadmat(test_keypoint_file)['keypoints3D'].astype(np.float32)
center_test = scio.loadmat(test_center_file)['centre_pixel'].astype(np.float32)

centre_test_world = pixel2world(center_test.copy(), fx, fy, u0, v0)

centerlefttop_test = centre_test_world.copy()
centerlefttop_test[:, 0, 0] = centerlefttop_test[:, 0, 0] - xy_thres
centerlefttop_test[:, 0, 1] = centerlefttop_test[:, 0, 1] + xy_thres

centerrightbottom_test = centre_test_world.copy()
centerrightbottom_test[:, 0, 0] = centerrightbottom_test[:, 0, 0] + xy_thres
centerrightbottom_test[:, 0, 1] = centerrightbottom_test[:, 0, 1] - xy_thres

test_lefttop_pixel = world2pixel(centerlefttop_test, fx, fy, u0, v0)
test_rightbottom_pixel = world2pixel(centerrightbottom_test, fx, fy, u0, v0)

keypointsUVD_train = scio.loadmat(train_keypoint_file)['keypoints3D'].astype(np.float32)
center_train = scio.loadmat(train_center_file)['centre_pixel'].astype(np.float32)
centre_train_world = pixel2world(center_train.copy(), fx, fy, u0, v0)

centerlefttop_train = centre_train_world.copy()
centerlefttop_train[:, 0, 0] = centerlefttop_train[:, 0, 0] - xy_thres
centerlefttop_train[:, 0, 1] = centerlefttop_train[:, 0, 1] + xy_thres

centerrightbottom_train = centre_train_world.copy()
centerrightbottom_train[:, 0, 0] = centerrightbottom_train[:, 0, 0] + xy_thres
centerrightbottom_train[:, 0, 1] = centerrightbottom_train[:, 0, 1] - xy_thres

train_lefttop_pixel = world2pixel(centerlefttop_train, fx, fy, u0, v0)
train_rightbottom_pixel = world2pixel(centerrightbottom_train, fx, fy, u0, v0)


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
        self.transformations = \
            transforms.Compose([transforms.RandomAffine(degrees=45, translate=(0.10, 0.10)),
                                transforms.RandomHorizontalFlip(p=0.1),
                                transforms.RandomVerticalFlip(p=0.1)])

    def __getitem__(self, index):
        # depth size: (480, 640), depth.max(): 2001, depth.min():733
        gesture_id = os.path.basename(os.path.dirname(self.depth_images[index]))
        person_id = os.path.basename(os.path.dirname(os.path.dirname(self.depth_images[index])))
        depth = np.loadtxt(self.depth_images[index], delimiter=',')
        label = int(gesture_id.split('G')[-1])
        center = pd.read_csv(
            self.ImgDir + 'Image/' + person_id + '/' + gesture_id + '/' + 'labels_' + person_id + '_' + gesture_id + '.csv',
            header=None)
        x_center, y_center = \
            center[center.iloc[:, 3].str.match(
                pathlib.Path(os.path.basename(self.depth_images[index])).stem + '.jpg')].iloc[:, [1, 2]].values[0]
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
        data = self.transformations(data)
        #plt.imsave('3.jpg', data.numpy().reshape((176, 176)))
        label = torch.from_numpy(label)
        label = label.long()
        label = label - 1
        return data, label

    def __len__(self):
        return len(self.depth_images)

def train():
    gesture_image_dataset = ntu_dataloader(dataset_dir, center_dir, depth_thres)
    train_samples = list(range(0, 1000, 2)) + list(range(201, 400, 2))
    test_samples = list(range(1, 200, 2))
    gesture_train_set = torch.utils.data.Subset(gesture_image_dataset, train_samples)
    gesture_test_set = torch.utils.data.Subset(gesture_image_dataset, test_samples)
    #gesture_train_set, gesture_test_set = torch.utils.data.random_split(gesture_image_dataset, (train_samples, test_samples))
    gesture_dataloader = torch.utils.data.DataLoader(gesture_train_set, batch_size=64, shuffle=True, num_workers=2)
    test_gesture_dataloader = torch.utils.data.DataLoader(gesture_test_set, batch_size=64, shuffle=False, num_workers=2)

    net = model.A2J_model(num_classes=keypointsNumber)
    net = net.cuda()
    post_precess = anchor.post_process(shape=[cropHeight // 16, cropWidth // 16], stride=16, P_h=None, P_w=None)
    criterion = anchor.A2J_loss(shape=[cropHeight // 16, cropWidth // 16], thres=[16.0, 32.0], stride=16, \
                                spatialFactor=spatialFactor, img_shape=[cropHeight, cropWidth], P_h=None, P_w=None)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=Weight_Decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                        filename=os.path.join(save_dir, 'train.log'), level=logging.INFO)
    logging.info('======================================================')
    criterion_2 = nn.CrossEntropyLoss()
    for epoch in range(nepoch):

        net = net.train()
        train_loss_add = 0.0
        Cls_loss_add = 0.0
        Reg_loss_add = 0.0
        Gesture_loss_add = 0.0
        training_running_corrects = 0.0
        testing_running_corrects = 0.0
        test_total_samples = 0.0
        total_samples = 0.0
        timer = time.time()

        # Training loop
        for i, (img, label) in enumerate(gesture_dataloader):

            torch.cuda.synchronize()

            img, label = img.cuda(), label.cuda()

            _, pred_gestures = net(img)

            # make_dot(heads, params=dict(list(net.named_parameters()))).render("multi_task_torchviz", format="png")

            # print(regression)
            optimizer.zero_grad()

            Gesture_loss = criterion_2(pred_gestures, label)
            _, predicted_gesture = torch.max(pred_gestures.data, 1)
            loss = Gesture_loss
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            total_samples = total_samples + len(label)

            training_running_corrects = training_running_corrects + (predicted_gesture == label).sum().item()

            Gesture_loss_add = Gesture_loss_add + (Gesture_loss.item()) * len(img)
            # printing loss info
            if i % 10 == 0:
                # print("BATCH INFORMATION")
                print('epoch: ', epoch, ' step: ', i,
                      'Gesture_loss', Gesture_loss.item(), ' total loss ', loss.item())

        scheduler.step(epoch)

        # time taken
        torch.cuda.synchronize()
        timer = time.time() - timer
        timer = timer / TrainImgFrames
        # print('==> time to learn 1 sample = %f (ms)' % (timer * 1000))
        print("EPOCH INFORMATION")
        print("Training running corrects epoch:", training_running_corrects)
        print("Total training samples", total_samples)
        epoch_train_acc = training_running_corrects / total_samples

        print('Gesture Accuracy:', epoch_train_acc)

        Gesture_loss_add = Gesture_loss_add / total_samples
        Error_test = 0
        Error_train = 0
        Error_test_wrist = 0

        if (epoch % 1 == 0):
            net = net.eval()
            output = torch.FloatTensor()
            outputTrain = torch.FloatTensor()

            for k, (gesture_img, gesture_label) in tqdm(enumerate(test_gesture_dataloader)):
                with torch.no_grad():
                    gesture_img, gesture_label = gesture_img.cuda(), gesture_label.cuda()
                    _, test_pred_gestures = net(gesture_img)
                    _, test_predicted_gesture = torch.max(test_pred_gestures.data, 1)
                    testing_running_corrects = testing_running_corrects + (test_predicted_gesture == gesture_label).sum().item()
                    test_total_samples = test_total_samples + len(gesture_label)
            print("Testing running corrects epoch:", testing_running_corrects)
            print("Total testing samples", test_total_samples)
            print("Test Gesture Accuracy", testing_running_corrects / test_total_samples)





if __name__ == '__main__':
    train()
    # test()
