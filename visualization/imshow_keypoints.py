import matplotlib.pyplot as plt
import numpy as np


def imshow_keypoints(image, keypoints, ground_truth_label=None, predicted_label=None):
    if len(keypoints.shape) != 2:
        raise Exception('Expected keypoints array to be (NumberOfKeypoints, 3)')
    if (type(image) or type(keypoints)) is not np.ndarray:
        raise Exception('Expected image and keypoint array to be numpy arrays')

    plt.rcParams["figure.figsize"] = (12, 8)
    fig = plt.figure()
    if (ground_truth_label and predicted_label) is not None:
        fig.suptitle("Ground truth label: " + str(ground_truth_label) + ' Predicted label: ' + str(predicted_label))

    line_width = 3
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(image, cmap="gray", vmin=image.min(), vmax=image.max())
    ax.set_title('2D skeleton NTU')
    ax.plot([keypoints[13, 1], keypoints[12, 1]], [keypoints[13, 0], keypoints[12, 0]],
            color=(153 / 255, 255 / 255, 204 / 255), linewidth=line_width)  # palm to wrist
    ax.plot([keypoints[13, 1], keypoints[11, 1]], [keypoints[13, 0], keypoints[11, 0]],
            color=(153 / 255, 255 / 255, 204 / 255), linewidth=line_width)  # palm to wrist back
    ax.plot([keypoints[13, 1], keypoints[10, 1]], [keypoints[13, 0], keypoints[10, 0]],
            color=(255 / 255, 153 / 255, 153 / 255), linewidth=line_width)  # palm to thumb root
    ax.plot([keypoints[9, 1], keypoints[10, 1]], [keypoints[9, 0], keypoints[10, 0]],
            color=(255 / 255, 102 / 255, 102 / 255), linewidth=line_width)  # thumb root to thumb mid
    ax.plot([keypoints[8, 1], keypoints[9, 1]], [keypoints[8, 0], keypoints[9, 0]],
            color=(255 / 255, 51 / 255, 51 / 255), linewidth=line_width)  # thumb mid to thumb tip
    ax.plot([keypoints[13, 1], keypoints[7, 1]], [keypoints[13, 0], keypoints[7, 0]],
            color=(153 / 255, 255 / 255, 153 / 255), linewidth=line_width)  # palm to index mid
    ax.plot([keypoints[6, 1], keypoints[7, 1]], [keypoints[6, 0], keypoints[7, 0]],
            color=(76.5 / 255, 255 / 255, 76.5 / 255), linewidth=line_width)  # index mid to index tip
    ax.plot([keypoints[13, 1], keypoints[5, 1]], [keypoints[13, 0], keypoints[5, 0]],
            color=(255 / 255, 204 / 255, 153 / 255), linewidth=line_width)  # palm to middle mid
    ax.plot([keypoints[4, 1], keypoints[5, 1]], [keypoints[4, 0], keypoints[5, 0]],
            color=(255 / 255, 165.5 / 255, 76.5 / 255), linewidth=line_width)  # middle mid to middle tip
    ax.plot([keypoints[13, 1], keypoints[3, 1]], [keypoints[13, 0], keypoints[3, 0]],
            color=(153 / 255, 204 / 255, 255 / 255), linewidth=line_width)  # palm to ring mid
    ax.plot([keypoints[2, 1], keypoints[3, 1]], [keypoints[2, 0], keypoints[3, 0]],
            color=(76.5 / 255, 165.5 / 255, 255 / 255), linewidth=line_width)  # ring mid to ring tip
    ax.plot([keypoints[13, 1], keypoints[1, 1]], [keypoints[13, 0], keypoints[1, 0]],
            color=(255 / 255, 153 / 255, 255 / 255), linewidth=line_width)  # palm to pinky mid
    ax.plot([keypoints[0, 1], keypoints[1, 1]], [keypoints[0, 0], keypoints[1, 0]],
            color=(255 / 255, 76.5 / 255, 255 / 255), linewidth=line_width)  # pinky mid to pinky tip

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_title('3D skeleton NTU')
    ax.invert_zaxis()
    ax.set_xlabel('X-axis')
    ax.set_zlabel('Y-axis')
    ax.set_ylabel('Depth')
    ax.plot([keypoints[13, 1], keypoints[12, 1]], [keypoints[13, 2], keypoints[12, 2]],
            [keypoints[13, 0], keypoints[12, 0]], color=(153 / 255, 255 / 255, 204 / 255),
            linewidth=line_width)  # palm to wrist
    ax.plot([keypoints[13, 1], keypoints[11, 1]], [keypoints[13, 2], keypoints[11, 2]],
            [keypoints[13, 0], keypoints[11, 0]], color=(153 / 255, 255 / 255, 204 / 255),
            linewidth=line_width)  # palm to wrist back
    ax.plot([keypoints[13, 1], keypoints[10, 1]], [keypoints[13, 2], keypoints[10, 2]],
            [keypoints[13, 0], keypoints[10, 0]], color=(255 / 255, 153 / 255, 153 / 255),
            linewidth=line_width)  # palm to thumb root
    ax.plot([keypoints[9, 1], keypoints[10, 1]], [keypoints[9, 2], keypoints[10, 2]],
            [keypoints[9, 0], keypoints[10, 0]], color=(255 / 255, 102 / 255, 102 / 255),
            linewidth=line_width)  # thumb root to thumb mid
    ax.plot([keypoints[8, 1], keypoints[9, 1]], [keypoints[8, 2], keypoints[9, 2]],
            [keypoints[8, 0], keypoints[9, 0]], color=(255 / 255, 51 / 255, 51 / 255),
            linewidth=line_width)  # thumb mid to thumb tip
    ax.plot([keypoints[13, 1], keypoints[7, 1]], [keypoints[13, 2], keypoints[7, 2]],
            [keypoints[13, 0], keypoints[7, 0]], color=(153 / 255, 255 / 255, 153 / 255),
            linewidth=line_width)  # palm to index mid
    ax.plot([keypoints[6, 1], keypoints[7, 1]], [keypoints[6, 2], keypoints[7, 2]],
            [keypoints[6, 0], keypoints[7, 0]], color=(76.5 / 255, 255 / 255, 76.5 / 255),
            linewidth=line_width)  # index mid to index tip
    ax.plot([keypoints[13, 1], keypoints[5, 1]], [keypoints[13, 2], keypoints[5, 2]],
            [keypoints[13, 0], keypoints[5, 0]], color=(255 / 255, 204 / 255, 153 / 255),
            linewidth=line_width)  # palm to middle mid
    ax.plot([keypoints[4, 1], keypoints[5, 1]], [keypoints[4, 2], keypoints[5, 2]],
            [keypoints[4, 0], keypoints[5, 0]], color=(255 / 255, 165.5 / 255, 76.5 / 255),
            linewidth=line_width)  # middle mid to middle tip
    ax.plot([keypoints[13, 1], keypoints[3, 1]], [keypoints[13, 2], keypoints[3, 2]],
            [keypoints[13, 0], keypoints[3, 0]], color=(153 / 255, 204 / 255, 255 / 255),
            linewidth=line_width)  # palm to ring mid
    ax.plot([keypoints[2, 1], keypoints[3, 1]], [keypoints[2, 2], keypoints[3, 2]],
            [keypoints[2, 0], keypoints[3, 0]], color=(76.5 / 255, 165.5 / 255, 255 / 255),
            linewidth=line_width)  # ring mid to ring tip
    ax.plot([keypoints[13, 1], keypoints[1, 1]], [keypoints[13, 2], keypoints[1, 2]],
            [keypoints[13, 0], keypoints[1, 0]], color=(255 / 255, 153 / 255, 255 / 255),
            linewidth=line_width)  # palm to pinky mid
    ax.plot([keypoints[0, 1], keypoints[1, 1]], [keypoints[0, 2], keypoints[1, 2]],
            [keypoints[0, 0], keypoints[1, 0]], color=(255 / 255, 76.5 / 255, 255 / 255),
            linewidth=line_width)  # pinky mid to pinky tip
