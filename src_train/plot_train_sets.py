import matplotlib.pyplot as plt
x_axis = [1, 5, 10, 20, 50, 60, 80]
gesture_accuracy = [30.1, 65.3, 79.2, 92.5, 94.1, 94.3, 96.2]
hand_pose_estimation = [9.54, 9.50, 9.54, 9.52, 9.48, 9.51, 9.46]

fig, ax = plt.subplots(nrows=1, ncols=2)
fig.figsize=(30,5)
ax[0].plot(x_axis, hand_pose_estimation)
ax[0].grid()
ax[0].set_xlabel('Proportion of dataset as training set (%)')
ax[0].set_ylabel('3D error')
ax[0].set_title('Hand pose estimation')


ax[1].plot(x_axis, gesture_accuracy)
ax[1].grid()
ax[1].set_xlabel('Proportion of dataset as training set (%)')
ax[1].set_ylabel('Accuracy (%)')
ax[1].set_title('Hand gesture')

plt.savefig('plot.jpg')