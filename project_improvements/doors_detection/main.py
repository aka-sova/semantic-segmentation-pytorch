

import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg



# img = cv2.imread('door_raw.jpg')

img2 = mpimg.imread('door_raw.jpg')

img2_hsv = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)

plt.imshow(img2)


# the yellow is approximately [193 165 65]
yellow_pixel = np.zeros((1, 1, 3))
yellow_pixel[0][0] = np.array([193, 165, 65])

yellow_pixel = yellow_pixel.astype(np.uint8)
yellow_pixel_hsv = cv2.cvtColor(yellow_pixel, cv2.COLOR_RGB2HSV)

# we receive:
# Hue 23
# Saturation 169
# Value 169

# We are interested in the Hue value. Define a margin of +- 10 and filter the original image

lower_yellow = np.array([20, 120, 50])
upper_yellow = np.array([30, 255, 255])

mask_img = cv2.inRange(img2_hsv, lower_yellow, upper_yellow)

plt.imshow(mask_img)
plt.savefig('Filteted_result.png')
