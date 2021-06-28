import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio


def normalize(x):
    return 255 * (np.array((x - np.min(x)) / (np.max(x) - np.min(x)))).astype('uint8')


testingImageDir = '../data/nyu/nyu_hand_dataset_v2/Preprossed/test_nyu/1.mat'
keypoint_file = '../data/nyu/nyu_keypointsUVD_test.mat'
keypointsUVD_test = scio.loadmat(keypoint_file)['keypoints3D'].astype(np.float32)
depth = scio.loadmat(testingImageDir)['depth']

plt.rcParams["figure.figsize"] = (12, 8)
fig = plt.figure()
line_width = 3
ax = fig.add_subplot(1, 2, 1)
ax.imshow(depth, cmap="gray", vmin=depth.min(), vmax=depth.max())
ax.set_title('2D skeleton NYU')
ax.plot([keypointsUVD_test[0, 13, 0], keypointsUVD_test[0, 12, 0]],
        [keypointsUVD_test[0, 13, 1], keypointsUVD_test[0, 12, 1]], color=(153 / 255, 255 / 255, 204 / 255),
        linewidth=line_width)  # palm to wrist
ax.plot([keypointsUVD_test[0, 13, 0], keypointsUVD_test[0, 11, 0]],
        [keypointsUVD_test[0, 13, 1], keypointsUVD_test[0, 11, 1]], color=(153 / 255, 255 / 255, 204 / 255),
        linewidth=line_width)  # palm to wrist back
ax.plot([keypointsUVD_test[0, 13, 0], keypointsUVD_test[0, 10, 0]],
        [keypointsUVD_test[0, 13, 1], keypointsUVD_test[0, 10, 1]], color=(255 / 255, 153 / 255, 153 / 255),
        linewidth=line_width)  # palm to thumb root
ax.plot([keypointsUVD_test[0, 9, 0], keypointsUVD_test[0, 10, 0]],
        [keypointsUVD_test[0, 9, 1], keypointsUVD_test[0, 10, 1]], color=(255 / 255, 102 / 255, 102 / 255),
        linewidth=line_width)  # thumb root to thumb mid
ax.plot([keypointsUVD_test[0, 8, 0], keypointsUVD_test[0, 9, 0]],
        [keypointsUVD_test[0, 8, 1], keypointsUVD_test[0, 9, 1]], color=(255 / 255, 51 / 255, 51 / 255),
        linewidth=line_width)  # thumb mid to thumb tip
ax.plot([keypointsUVD_test[0, 13, 0], keypointsUVD_test[0, 7, 0]],
        [keypointsUVD_test[0, 13, 1], keypointsUVD_test[0, 7, 1]], color=(153 / 255, 255 / 255, 153 / 255),
        linewidth=line_width)  # palm to index mid
ax.plot([keypointsUVD_test[0, 6, 0], keypointsUVD_test[0, 7, 0]],
        [keypointsUVD_test[0, 6, 1], keypointsUVD_test[0, 7, 1]], color=(76.5 / 255, 255 / 255, 76.5 / 255),
        linewidth=line_width)  # index mid to index tip
ax.plot([keypointsUVD_test[0, 13, 0], keypointsUVD_test[0, 5, 0]],
        [keypointsUVD_test[0, 13, 1], keypointsUVD_test[0, 5, 1]], color=(255 / 255, 204 / 255, 153 / 255),
        linewidth=line_width)  # palm to middle mid
ax.plot([keypointsUVD_test[0, 4, 0], keypointsUVD_test[0, 5, 0]],
        [keypointsUVD_test[0, 4, 1], keypointsUVD_test[0, 5, 1]], color=(255 / 255, 165.5 / 255, 76.5 / 255),
        linewidth=line_width)  # middle mid to middle tip
ax.plot([keypointsUVD_test[0, 13, 0], keypointsUVD_test[0, 3, 0]],
        [keypointsUVD_test[0, 13, 1], keypointsUVD_test[0, 3, 1]], color=(153 / 255, 204 / 255, 255 / 255),
        linewidth=line_width)  # palm to ring mid
ax.plot([keypointsUVD_test[0, 2, 0], keypointsUVD_test[0, 3, 0]],
        [keypointsUVD_test[0, 2, 1], keypointsUVD_test[0, 3, 1]], color=(76.5 / 255, 165.5 / 255, 255 / 255),
        linewidth=line_width)  # ring mid to ring tip
ax.plot([keypointsUVD_test[0, 13, 0], keypointsUVD_test[0, 1, 0]],
        [keypointsUVD_test[0, 13, 1], keypointsUVD_test[0, 1, 1]], color=(255 / 255, 153 / 255, 255 / 255),
        linewidth=line_width)  # palm to pinky mid
ax.plot([keypointsUVD_test[0, 0, 0], keypointsUVD_test[0, 1, 0]],
        [keypointsUVD_test[0, 0, 1], keypointsUVD_test[0, 1, 1]], color=(255 / 255, 76.5 / 255, 255 / 255),
        linewidth=line_width)  # pinky mid to pinky tip

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.invert_zaxis()
ax.set_xlabel('X-axis')
ax.set_zlabel('Y-axis')
ax.set_ylabel('Depth')
ax.set_title('3D skeleton NYU')
ax.plot([keypointsUVD_test[0, 13, 0], keypointsUVD_test[0, 12, 0]],
        [keypointsUVD_test[0, 13, 2], keypointsUVD_test[0, 12, 2]],
        [keypointsUVD_test[0, 13, 1], keypointsUVD_test[0, 12, 1]], color=(153 / 255, 255 / 255, 204 / 255),
        linewidth=line_width)  # palm to wrist
ax.plot([keypointsUVD_test[0, 13, 0], keypointsUVD_test[0, 11, 0]],
        [keypointsUVD_test[0, 13, 2], keypointsUVD_test[0, 11, 2]],
        [keypointsUVD_test[0, 13, 1], keypointsUVD_test[0, 11, 1]], color=(153 / 255, 255 / 255, 204 / 255),
        linewidth=line_width)  # palm to wrist back
ax.plot([keypointsUVD_test[0, 13, 0], keypointsUVD_test[0, 10, 0]],
        [keypointsUVD_test[0, 13, 2], keypointsUVD_test[0, 10, 2]],
        [keypointsUVD_test[0, 13, 1], keypointsUVD_test[0, 10, 1]], color=(255 / 255, 153 / 255, 153 / 255),
        linewidth=line_width)  # palm to thumb root
ax.plot([keypointsUVD_test[0, 9, 0], keypointsUVD_test[0, 10, 0]],
        [keypointsUVD_test[0, 9, 2], keypointsUVD_test[0, 10, 2]],
        [keypointsUVD_test[0, 9, 1], keypointsUVD_test[0, 10, 1]], color=(255 / 255, 102 / 255, 102 / 255),
        linewidth=line_width)  # thumb root to thumb mid
ax.plot([keypointsUVD_test[0, 8, 0], keypointsUVD_test[0, 9, 0]],
        [keypointsUVD_test[0, 8, 2], keypointsUVD_test[0, 9, 2]],
        [keypointsUVD_test[0, 8, 1], keypointsUVD_test[0, 9, 1]], color=(255 / 255, 51 / 255, 51 / 255),
        linewidth=line_width)  # thumb mid to thumb tip
ax.plot([keypointsUVD_test[0, 13, 0], keypointsUVD_test[0, 7, 0]],
        [keypointsUVD_test[0, 13, 2], keypointsUVD_test[0, 7, 2]],
        [keypointsUVD_test[0, 13, 1], keypointsUVD_test[0, 7, 1]], color=(153 / 255, 255 / 255, 153 / 255),
        linewidth=line_width)  # palm to index mid
ax.plot([keypointsUVD_test[0, 6, 0], keypointsUVD_test[0, 7, 0]],
        [keypointsUVD_test[0, 6, 2], keypointsUVD_test[0, 7, 2]],
        [keypointsUVD_test[0, 6, 1], keypointsUVD_test[0, 7, 1]], color=(76.5 / 255, 255 / 255, 76.5 / 255),
        linewidth=line_width)  # index mid to index tip
ax.plot([keypointsUVD_test[0, 13, 0], keypointsUVD_test[0, 5, 0]],
        [keypointsUVD_test[0, 13, 2], keypointsUVD_test[0, 5, 2]],
        [keypointsUVD_test[0, 13, 1], keypointsUVD_test[0, 5, 1]], color=(255 / 255, 204 / 255, 153 / 255),
        linewidth=line_width)  # palm to middle mid
ax.plot([keypointsUVD_test[0, 4, 0], keypointsUVD_test[0, 5, 0]],
        [keypointsUVD_test[0, 4, 2], keypointsUVD_test[0, 5, 2]],
        [keypointsUVD_test[0, 4, 1], keypointsUVD_test[0, 5, 1]], color=(255 / 255, 165.5 / 255, 76.5 / 255),
        linewidth=line_width)  # middle mid to middle tip
ax.plot([keypointsUVD_test[0, 13, 0], keypointsUVD_test[0, 3, 0]],
        [keypointsUVD_test[0, 13, 2], keypointsUVD_test[0, 3, 2]],
        [keypointsUVD_test[0, 13, 1], keypointsUVD_test[0, 3, 1]], color=(153 / 255, 204 / 255, 255 / 255),
        linewidth=line_width)  # palm to ring mid
ax.plot([keypointsUVD_test[0, 2, 0], keypointsUVD_test[0, 3, 0]],
        [keypointsUVD_test[0, 2, 2], keypointsUVD_test[0, 3, 2]],
        [keypointsUVD_test[0, 2, 1], keypointsUVD_test[0, 3, 1]], color=(76.5 / 255, 165.5 / 255, 255 / 255),
        linewidth=line_width)  # ring mid to ring tip
ax.plot([keypointsUVD_test[0, 13, 0], keypointsUVD_test[0, 1, 0]],
        [keypointsUVD_test[0, 13, 2], keypointsUVD_test[0, 1, 2]],
        [keypointsUVD_test[0, 13, 1], keypointsUVD_test[0, 1, 1]], color=(255 / 255, 153 / 255, 255 / 255),
        linewidth=line_width)  # palm to pinky mid
ax.plot([keypointsUVD_test[0, 0, 0], keypointsUVD_test[0, 1, 0]],
        [keypointsUVD_test[0, 0, 2], keypointsUVD_test[0, 1, 2]],
        [keypointsUVD_test[0, 0, 1], keypointsUVD_test[0, 1, 1]], color=(255 / 255, 76.5 / 255, 255 / 255),
        linewidth=line_width)  # pinky mid to pinky tip
