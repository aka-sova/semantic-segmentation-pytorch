

import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os
from sklearn import linear_model, datasets

from utils import find_recursive
from project_improvements.doors_detection.scripts import get_segmentation

cur_cwd = os.getcwd()
# os.chdir(os.path.abspath(os.path.join(cur_cwd, 'project_improvements', 'doors_detection')))



# get all the files in the input folder

input_folder = os.path.join(os.path.join(cur_cwd, 'input'))
output_folder = os.path.join(os.path.join(cur_cwd, 'output'))
final_folder = os.path.join(os.path.join(output_folder, 'final'))

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

if not os.path.exists(final_folder):
    os.mkdir(final_folder)


input_imgs = find_recursive(input_folder, ext=['.png', '.jpg'])

for input_img in input_imgs:
    print("Inferring for " + input_img)
    img_name = os.path.basename(input_img)[:-4] # remove the extension
    debug_folder = os.path.join(os.path.join(output_folder, img_name))
    segmented_img = get_segmentation(input_img, debug=True, debug_folder=debug_folder)
    if segmented_img is not None:
        cv2.imwrite(os.path.join(final_folder, img_name + '.jpg'), segmented_img)



