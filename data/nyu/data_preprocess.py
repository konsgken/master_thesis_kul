import os
import scipy.io as scio
import cv2
dataset_dir = './nyu_hand_dataset_v2/train/'
save_dir = './nyu_hand_dataset_v2/Preprossed/train_nyu/'
tot_frame_num = 72757

kinect_index = 1

for image_index in range(1, tot_frame_num + 1):
    filename_prefix = '%d_%07d' % (kinect_index, image_index)
    if os.path.exists(dataset_dir + 'depth_' + filename_prefix + '.png'):
        # The top 8 bits of depth are packed into green and the lower 8 bits into blue.
        depth = cv2.imread(dataset_dir + 'depth_' + filename_prefix + '.png')
        depth = depth[:, :, 0].astype('uint16') + (depth[:, :, 1].astype('uint16') << 8)
        scio.savemat(save_dir + str(image_index) + '.mat', {'depth': depth})
        print(save_dir + str(image_index) + '.mat')