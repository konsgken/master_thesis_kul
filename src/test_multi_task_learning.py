import cv2
import torch
import torch.utils.data
import numpy as np
import os
import src_train.resnet18_group_normalization_model as model
import anchor as anchor
import random
import pathlib
import pandas as pd
import glob
import visualization.imshow_keypoints

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# DataHyperParms
TestImgFrames = 8252
keypointsNumber = 14
cropWidth = 176
cropHeight = 176

xy_thres = 110
depth_thres = 150

randomseed = 12345
random.seed(randomseed)
np.random.seed(randomseed)
torch.manual_seed(randomseed)
import matplotlib.pyplot as plt

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
model_dir = '../best_models/net_31_wetD_0.0001_depFact_0.5_RegFact_3_rndShft_5.pth'
result_file = 'result_NYU.txt'

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
        label = torch.from_numpy(label)
        label = label.long()
        label = label - 1
        return data, label

    def __len__(self):
        return len(self.depth_images)


gesture_image_dataset = ntu_dataloader(dataset_dir, center_dir, depth_thres)
print("Training Samples NTU: ", len(gesture_image_dataset))
gesture_dataloader = torch.utils.data.DataLoader(gesture_image_dataset, batch_size=1, shuffle=True, num_workers=4)
net = model.A2J_model(num_classes=keypointsNumber)
net.load_state_dict(torch.load(model_dir))
net = net.cuda()
net.eval()

post_precess = anchor.post_process(shape=[cropHeight // 16, cropWidth // 16], stride=16, P_h=None, P_w=None)

output = torch.FloatTensor()
torch.cuda.synchronize()
for i, (img, label) in enumerate(gesture_dataloader):
    if i >= 5:
        break
    with torch.no_grad():
        img, label = img.cuda(), label.cuda()
        heads, gestures = net(img)
        _, predicted_gesture = torch.max(gestures.data, 1)
        pred_keypoints = post_precess(heads, voting=False)
        visualization.imshow_keypoints(img.cpu().data.numpy().squeeze(), pred_keypoints.cpu().data.numpy().squeeze(),
                                       ground_truth_label=int(label.cpu().data.numpy()),
                                       predicted_label=int(predicted_gesture.cpu().data.numpy()))
