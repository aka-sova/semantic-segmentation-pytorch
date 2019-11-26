

import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os
from sklearn import linear_model, datasets


cur_cwd = os.getcwd()
os.chdir(os.path.abspath(os.path.join(cur_cwd, 'project_improvements', 'doors_detection')))


# img = cv2.imread('door_raw.jpg')

img2 = mpimg.imread('door_raw.jpg')
img2 = cv2.resize(img2, None, fx=0.2, fy=0.2)

img2_hsv = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)

plt.imshow(img2)

# get the yellow pixel in hsv
# the yellow is approximately [193 165 65]  in RGB
yellow_pixel = np.zeros((1, 1, 3))
yellow_pixel[0][0] = np.array([193, 165, 65])
yellow_pixel = yellow_pixel.astype(np.uint8)

yellow_pixel_hsv = cv2.cvtColor(yellow_pixel, cv2.COLOR_RGB2HSV)
# the yellow is approximately [23 169 169]  in HSV

# We are interested mainly in the Hue value, which is quiet saturated. Define a margin of +- 10 and filter the original image
lower_yellow = np.array([20, 120, 50])
upper_yellow = np.array([30, 255, 255])

mask_img = cv2.inRange(img2_hsv, lower_yellow, upper_yellow)

plt.imshow(mask_img, 'gray')
plt.savefig('Filteted_result.png')


# Using the RANSAC algorithm to find the lines which resemble the lines of the doors
# Regular algorithms find only one model using the RANSAC algorithm.
# We here have to find numerous lines, which will sum to 3 after the non-max suppression.

# extract the array of X and Y location of all 'white' pixels in the mask image
x_coords = []
y_coords = []

for row_num in range(mask_img.shape[0]):
    for col_num in range(mask_img.shape[1]):
        if mask_img[row_num][col_num] == 255:
            x_coords.append(col_num)
            y_coords.append(row_num)

x_coords = np.reshape(x_coords, (-1, 1))
y_coords = np.reshape(y_coords, (-1, 1))

ransac_line = linear_model.RANSACRegressor(residual_threshold=10)
ransac_line.fit(x_coords, y_coords)

# draw the line on the image
line_X = np.arange(x_coords.min(), x_coords.max())[:, np.newaxis]
line_y_ransac = ransac_line.predict(line_X)


plt.plot(line_X, line_y_ransac , color='cornflowerblue', linewidth=2,
         label='RANSAC regressor')
plt.savefig('RANSAC_example.png')