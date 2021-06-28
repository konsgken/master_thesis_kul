import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import torch
import model as model
import anchor as anchor

keypointsNumber = 14
plt.close('all')
depth_thres = 100
MEAN = np.load('../data/nyu/nyu_mean.npy')
STD = np.load('../data/nyu/nyu_std.npy')
model_dir = '../model/NYU.pth'
img = np.loadtxt('../data/NTU/Depth/P1/G2/10.txt', delimiter=',')
plt.rcParams["figure.figsize"] = (12, 8)
fig = plt.figure()
ax = fig.add_subplot(1, 3, 1)
ax.imshow(img, cmap='gray', vmin=img.min(), vmax=img.max())
ax.set_title('Initial Depth Image')
x = 270
y = 170
cropWidth = 176
cropHeight = 176
cropped_img = img[x: x + 176, y: y + 176]
plot_img = cropped_img.copy()
ax = fig.add_subplot(1, 3, 2)
ax.imshow(cropped_img, cmap='gray', vmin=cropped_img.min(), vmax=cropped_img.max())
ax.set_title('Cropped Image')
center_cropped_img = cropped_img[int(cropped_img.shape[0]/2), int(cropped_img.shape[1]/2)]

cropped_img = np.asarray(cropped_img, dtype='float32')
cropped_img[np.where(cropped_img >= center_cropped_img + depth_thres)] = center_cropped_img
cropped_img[np.where(cropped_img <= center_cropped_img - depth_thres)] = center_cropped_img
cropped_img = cropped_img - center_cropped_img
cropped_img = (cropped_img - MEAN) / STD
ax = fig.add_subplot(1, 3, 3)
ax.imshow(cropped_img, cmap='gray', vmin=cropped_img.min(), vmax=cropped_img.max())
ax.set_title('Subtract center depth value and normalize')
image = torch.from_numpy(cropped_img)
image = torch.autograd.Variable(image, requires_grad=True)
image = image.unsqueeze(0)
image = image.unsqueeze(0)


net = model.A2J_model(num_classes=keypointsNumber)
net.load_state_dict(torch.load(model_dir))
net = net.cuda()
net.eval()
post_precess = anchor.post_process(shape=[cropHeight//16, cropWidth//16],stride=16,P_h=None, P_w=None)
with torch.no_grad():
    image = image.cuda()
    heads = net(image)
    pred_keypoints = post_precess(heads, voting=False)

keypointsUVD_test = pred_keypoints.cpu().data.numpy()

keypointsUVD_test = keypointsUVD_test +[x, y, center_cropped_img]


plt.rcParams["figure.figsize"] = (12,8)
fig = plt.figure()
line_width = 3
ax = fig.add_subplot(1, 2, 1)
ax.imshow(img, cmap="gray", vmin=img.min(), vmax=img.max())
ax.set_title('2D skeleton NTU')
ax.plot([keypointsUVD_test[0, 13, 1], keypointsUVD_test[0, 12, 1]], [keypointsUVD_test[0, 13, 0], keypointsUVD_test[0, 12, 0]], color=(153/255, 255/255, 204/255), linewidth=line_width) #palm to wrist
ax.plot([keypointsUVD_test[0, 13, 1], keypointsUVD_test[0, 11, 1]], [keypointsUVD_test[0, 13, 0], keypointsUVD_test[0, 11, 0]], color=(153/255, 255/255, 204/255), linewidth=line_width) #palm to wrist back
ax.plot([keypointsUVD_test[0, 13, 1], keypointsUVD_test[0, 10, 1]], [keypointsUVD_test[0, 13, 0], keypointsUVD_test[0, 10, 0]], color=(255/255,153/255,153/255), linewidth=line_width) #palm to thumb root
ax.plot([keypointsUVD_test[0, 9, 1], keypointsUVD_test[0, 10, 1]], [keypointsUVD_test[0, 9, 0], keypointsUVD_test[0, 10, 0]], color=(255/255,102/255,102/255), linewidth=line_width) #thumb root to thumb mid
ax.plot([keypointsUVD_test[0, 8, 1], keypointsUVD_test[0, 9, 1]], [keypointsUVD_test[0, 8, 0], keypointsUVD_test[0, 9, 0]], color=(255/255,51/255,51/255), linewidth=line_width) #thumb mid to thumb tip
ax.plot([keypointsUVD_test[0, 13, 1], keypointsUVD_test[0, 7, 1]], [keypointsUVD_test[0, 13, 0], keypointsUVD_test[0, 7, 0]], color=(153/255,255/255,153/255), linewidth=line_width) #palm to index mid
ax.plot([keypointsUVD_test[0, 6, 1], keypointsUVD_test[0, 7, 1]], [keypointsUVD_test[0, 6, 0], keypointsUVD_test[0, 7, 0]], color=(76.5/255,255/255,76.5/255), linewidth=line_width) #index mid to index tip
ax.plot([keypointsUVD_test[0, 13, 1], keypointsUVD_test[0, 5, 1]], [keypointsUVD_test[0, 13, 0], keypointsUVD_test[0, 5, 0]], color=(255/255,204/255,153/255), linewidth=line_width) #palm to middle mid
ax.plot([keypointsUVD_test[0, 4, 1], keypointsUVD_test[0, 5, 1]], [keypointsUVD_test[0, 4, 0], keypointsUVD_test[0, 5, 0]], color=(255/255,165.5/255,76.5/255), linewidth=line_width) #middle mid to middle tip
ax.plot([keypointsUVD_test[0, 13, 1], keypointsUVD_test[0, 3, 1]], [keypointsUVD_test[0, 13, 0], keypointsUVD_test[0, 3, 0]], color=(153/255,204/255,255/255), linewidth=line_width) #palm to ring mid
ax.plot([keypointsUVD_test[0, 2, 1], keypointsUVD_test[0, 3, 1]], [keypointsUVD_test[0, 2, 0], keypointsUVD_test[0, 3, 0]], color=(76.5/255,165.5/255,255/255), linewidth=line_width) #ring mid to ring tip
ax.plot([keypointsUVD_test[0, 13, 1], keypointsUVD_test[0, 1, 1]], [keypointsUVD_test[0, 13, 0], keypointsUVD_test[0, 1, 0]], color=(255/255,153/255,255/255), linewidth=line_width) #palm to pinky mid
ax.plot([keypointsUVD_test[0, 0, 1], keypointsUVD_test[0, 1, 1]], [keypointsUVD_test[0, 0, 0], keypointsUVD_test[0, 1, 0]], color=(255/255,76.5/255,255/255), linewidth=line_width) #pinky mid to pinky tip

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.set_title('3D skeleton NTU')
ax.invert_zaxis()
ax.set_xlabel('X-axis')
ax.set_zlabel('Y-axis')
ax.set_ylabel('Depth')
ax.plot([keypointsUVD_test[0, 13, 1], keypointsUVD_test[0, 12, 1]], [keypointsUVD_test[0, 13, 2], keypointsUVD_test[0, 12, 2]], [keypointsUVD_test[0, 13, 0], keypointsUVD_test[0, 12, 0]], color=(153/255, 255/255, 204/255), linewidth=line_width) #palm to wrist
ax.plot([keypointsUVD_test[0, 13, 1], keypointsUVD_test[0, 11, 1]], [keypointsUVD_test[0, 13, 2], keypointsUVD_test[0, 11, 2]], [keypointsUVD_test[0, 13, 0], keypointsUVD_test[0, 11, 0]],color=(153/255, 255/255, 204/255), linewidth=line_width) #palm to wrist back
ax.plot([keypointsUVD_test[0, 13, 1], keypointsUVD_test[0, 10, 1]], [keypointsUVD_test[0, 13, 2], keypointsUVD_test[0, 10, 2]], [keypointsUVD_test[0, 13, 0], keypointsUVD_test[0, 10, 0]],color=(255/255,153/255,153/255), linewidth=line_width) #palm to thumb root
ax.plot([keypointsUVD_test[0, 9, 1], keypointsUVD_test[0, 10, 1]], [keypointsUVD_test[0, 9, 2], keypointsUVD_test[0, 10, 2]], [keypointsUVD_test[0, 9, 0], keypointsUVD_test[0, 10, 0]],color=(255/255,102/255,102/255), linewidth=line_width) #thumb root to thumb mid
ax.plot([keypointsUVD_test[0, 8, 1], keypointsUVD_test[0, 9, 1]], [keypointsUVD_test[0, 8, 2], keypointsUVD_test[0, 9, 2]], [keypointsUVD_test[0, 8, 0], keypointsUVD_test[0, 9, 0]],color=(255/255,51/255,51/255), linewidth=line_width) #thumb mid to thumb tip
ax.plot([keypointsUVD_test[0, 13, 1], keypointsUVD_test[0, 7, 1]], [keypointsUVD_test[0, 13, 2], keypointsUVD_test[0, 7, 2]], [keypointsUVD_test[0, 13, 0], keypointsUVD_test[0, 7, 0]],color=(153/255,255/255,153/255), linewidth=line_width) #palm to index mid
ax.plot([keypointsUVD_test[0, 6, 1], keypointsUVD_test[0, 7, 1]], [keypointsUVD_test[0, 6, 2], keypointsUVD_test[0, 7, 2]], [keypointsUVD_test[0, 6, 0], keypointsUVD_test[0, 7, 0]],color=(76.5/255,255/255,76.5/255), linewidth=line_width) #index mid to index tip
ax.plot([keypointsUVD_test[0, 13, 1], keypointsUVD_test[0, 5, 1]], [keypointsUVD_test[0, 13, 2], keypointsUVD_test[0, 5, 2]], [keypointsUVD_test[0, 13, 0], keypointsUVD_test[0, 5, 0]],color=(255/255,204/255,153/255), linewidth=line_width) #palm to middle mid
ax.plot([keypointsUVD_test[0, 4, 1], keypointsUVD_test[0, 5, 1]], [keypointsUVD_test[0, 4, 2], keypointsUVD_test[0, 5, 2]], [keypointsUVD_test[0, 4, 0], keypointsUVD_test[0, 5, 0]],color=(255/255,165.5/255,76.5/255), linewidth=line_width) #middle mid to middle tip
ax.plot([keypointsUVD_test[0, 13, 1], keypointsUVD_test[0, 3, 1]], [keypointsUVD_test[0, 13, 2], keypointsUVD_test[0, 3, 2]], [keypointsUVD_test[0, 13, 0], keypointsUVD_test[0, 3, 0]],color=(153/255,204/255,255/255), linewidth=line_width) #palm to ring mid
ax.plot([keypointsUVD_test[0, 2, 1], keypointsUVD_test[0, 3, 1]], [keypointsUVD_test[0, 2, 2], keypointsUVD_test[0, 3, 2]], [keypointsUVD_test[0, 2, 0], keypointsUVD_test[0, 3, 0]],color=(76.5/255,165.5/255,255/255), linewidth=line_width) #ring mid to ring tip
ax.plot([keypointsUVD_test[0, 13, 1], keypointsUVD_test[0, 1, 1]], [keypointsUVD_test[0, 13, 2], keypointsUVD_test[0, 1, 2]], [keypointsUVD_test[0, 13, 0], keypointsUVD_test[0, 1, 0]],color=(255/255,153/255,255/255), linewidth=line_width) #palm to pinky mid
ax.plot([keypointsUVD_test[0, 0, 1], keypointsUVD_test[0, 1, 1]], [keypointsUVD_test[0, 0, 2], keypointsUVD_test[0, 1, 2]], [keypointsUVD_test[0, 0, 0], keypointsUVD_test[0, 1, 0]],color=(255/255,76.5/255,255/255), linewidth=line_width) #pinky mid to pinky tip

# def imshow_keypoints(image, keypoints, ground_truth_label=None, predicted_label=None):
#     plt.rcParams["figure.figsize"] = (12, 8)
#     fig = plt.figure()
#     line_width = 3
#     ax = fig.add_subplot(1, 2, 1)
#     ax.imshow(image, cmap="gray", vmin=image.min(), vmax=image.max())
#     ax.set_title('2D skeleton NTU')
#     ax.plot([keypoints[13, 1], keypoints[12, 1]], [keypoints[13, 0], keypoints[12, 0]],
#             color=(153 / 255, 255 / 255, 204 / 255), linewidth=line_width)  # palm to wrist
#     ax.plot([keypoints[13, 1], keypoints[11, 1]], [keypoints[13, 0], keypoints[11, 0]],
#             color=(153 / 255, 255 / 255, 204 / 255), linewidth=line_width)  # palm to wrist back
#     ax.plot([keypoints[13, 1], keypoints[10, 1]], [keypoints[13, 0], keypoints[10, 0]],
#             color=(255 / 255, 153 / 255, 153 / 255), linewidth=line_width)  # palm to thumb root
#     ax.plot([keypoints[9, 1], keypoints[10, 1]], [keypoints[9, 0], keypoints[10, 0]],
#             color=(255 / 255, 102 / 255, 102 / 255), linewidth=line_width)  # thumb root to thumb mid
#     ax.plot([keypoints[8, 1], keypoints[9, 1]], [keypoints[8, 0], keypoints[9, 0]],
#             color=(255 / 255, 51 / 255, 51 / 255), linewidth=line_width)  # thumb mid to thumb tip
#     ax.plot([keypoints[13, 1], keypoints[7, 1]], [keypoints[13, 0], keypoints[7, 0]],
#             color=(153 / 255, 255 / 255, 153 / 255), linewidth=line_width)  # palm to index mid
#     ax.plot([keypoints[6, 1], keypoints[7, 1]], [keypoints[6, 0], keypoints[7, 0]],
#             color=(76.5 / 255, 255 / 255, 76.5 / 255), linewidth=line_width)  # index mid to index tip
#     ax.plot([keypoints[13, 1], keypoints[5, 1]], [keypoints[13, 0], keypoints[5, 0]],
#             color=(255 / 255, 204 / 255, 153 / 255), linewidth=line_width)  # palm to middle mid
#     ax.plot([keypoints[4, 1], keypoints[5, 1]], [keypoints[4, 0], keypoints[5, 0]],
#             color=(255 / 255, 165.5 / 255, 76.5 / 255), linewidth=line_width)  # middle mid to middle tip
#     ax.plot([keypoints[13, 1], keypoints[3, 1]], [keypoints[13, 0], keypoints[3, 0]],
#             color=(153 / 255, 204 / 255, 255 / 255), linewidth=line_width)  # palm to ring mid
#     ax.plot([keypoints[2, 1], keypoints[3, 1]], [keypoints[2, 0], keypoints[3, 0]],
#             color=(76.5 / 255, 165.5 / 255, 255 / 255), linewidth=line_width)  # ring mid to ring tip
#     ax.plot([keypoints[13, 1], keypoints[1, 1]], [keypoints[13, 0], keypoints[1, 0]],
#             color=(255 / 255, 153 / 255, 255 / 255), linewidth=line_width)  # palm to pinky mid
#     ax.plot([keypoints[0, 1], keypoints[1, 1]], [keypoints[0, 0], keypoints[1, 0]],
#             color=(255 / 255, 76.5 / 255, 255 / 255), linewidth=line_width)  # pinky mid to pinky tip
#
#     ax = fig.add_subplot(1, 2, 2, projection='3d')
#     ax.set_title('3D skeleton NTU')
#     ax.invert_zaxis()
#     ax.set_xlabel('X-axis')
#     ax.set_zlabel('Y-axis')
#     ax.set_ylabel('Depth')
#     ax.plot([keypoints[13, 1], keypoints[12, 1]], [keypoints[13, 2], keypoints[12, 2]],
#             [keypoints[13, 0], keypoints[12, 0]], color=(153 / 255, 255 / 255, 204 / 255),
#             linewidth=line_width)  # palm to wrist
#     ax.plot([keypoints[13, 1], keypoints[11, 1]], [keypoints[13, 2], keypoints[11, 2]],
#             [keypoints[13, 0], keypoints[11, 0]], color=(153 / 255, 255 / 255, 204 / 255),
#             linewidth=line_width)  # palm to wrist back
#     ax.plot([keypoints[13, 1], keypoints[10, 1]], [keypoints[13, 2], keypoints[10, 2]],
#             [keypoints[13, 0], keypoints[10, 0]], color=(255 / 255, 153 / 255, 153 / 255),
#             linewidth=line_width)  # palm to thumb root
#     ax.plot([keypoints[9, 1], keypoints[10, 1]], [keypoints[9, 2], keypoints[10, 2]],
#             [keypoints[9, 0], keypoints[10, 0]], color=(255 / 255, 102 / 255, 102 / 255),
#             linewidth=line_width)  # thumb root to thumb mid
#     ax.plot([keypoints[8, 1], keypoints[9, 1]], [keypoints[8, 2], keypoints[9, 2]],
#             [keypoints[8, 0], keypoints[9, 0]], color=(255 / 255, 51 / 255, 51 / 255),
#             linewidth=line_width)  # thumb mid to thumb tip
#     ax.plot([keypoints[13, 1], keypoints[7, 1]], [keypoints[13, 2], keypoints[7, 2]],
#             [keypoints[13, 0], keypoints[7, 0]], color=(153 / 255, 255 / 255, 153 / 255),
#             linewidth=line_width)  # palm to index mid
#     ax.plot([keypoints[6, 1], keypoints[7, 1]], [keypoints[6, 2], keypoints[7, 2]],
#             [keypoints[6, 0], keypoints[7, 0]], color=(76.5 / 255, 255 / 255, 76.5 / 255),
#             linewidth=line_width)  # index mid to index tip
#     ax.plot([keypoints[13, 1], keypoints[5, 1]], [keypoints[13, 2], keypoints[5, 2]],
#             [keypoints[13, 0], keypoints[5, 0]], color=(255 / 255, 204 / 255, 153 / 255),
#             linewidth=line_width)  # palm to middle mid
#     ax.plot([keypoints[4, 1], keypoints[5, 1]], [keypoints[4, 2], keypoints[5, 2]],
#             [keypoints[4, 0], keypoints[5, 0]], color=(255 / 255, 165.5 / 255, 76.5 / 255),
#             linewidth=line_width)  # middle mid to middle tip
#     ax.plot([keypoints[13, 1], keypoints[3, 1]], [keypoints[13, 2], keypoints[3, 2]],
#             [keypoints[13, 0], keypoints[3, 0]], color=(153 / 255, 204 / 255, 255 / 255),
#             linewidth=line_width)  # palm to ring mid
#     ax.plot([keypoints[2, 1], keypoints[3, 1]], [keypoints[2, 2], keypoints[3, 2]],
#             [keypoints[2, 0], keypoints[3, 0]], color=(76.5 / 255, 165.5 / 255, 255 / 255),
#             linewidth=line_width)  # ring mid to ring tip
#     ax.plot([keypoints[13, 1], keypoints[1, 1]], [keypoints[13, 2], keypoints[1, 2]],
#             [keypoints[13, 0], keypoints[1, 0]], color=(255 / 255, 153 / 255, 255 / 255),
#             linewidth=line_width)  # palm to pinky mid
#     ax.plot([keypoints[0, 1], keypoints[1, 1]], [keypoints[0, 2], keypoints[1, 2]],
#             [keypoints[0, 0], keypoints[1, 0]], color=(255 / 255, 76.5 / 255, 255 / 255),
#             linewidth=line_width)  # pinky mid to pinky tip
#
# keypoints = keypointsUVD_test.squeeze()
# imshow_keypoints(img, keypoints)