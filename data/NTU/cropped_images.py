import glob
import os
import cv2
import pandas as pd
images = glob.glob('Image/**/**/*.jpg')
labels = [int(os.path.basename(os.path.dirname(i)).split('G')[-1]) for i in images]
h, w = 176, 176
for idx, image_path in enumerate(images):
    image = cv2.imread(image_path)
    center = pd.read_csv(glob.glob(os.path.dirname(image_path)+'/*.csv')[0], header=None)
    x_center, y_center = center[center.iloc[:,3].str.match(os.path.basename(image_path))].iloc[:,[1,2]].values[0]
    crop_img = image[int(y_center - h/2): int(y_center + h/2), int(x_center - w/2): int(x_center + w/2)]
    cv2.imwrite('Cropped_Images/'+str(idx)+'.jpg',crop_img)