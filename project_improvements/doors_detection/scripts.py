import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os
from sklearn import linear_model, datasets
from collections import defaultdict


def rgb_2_hsv_pixel(rgb: list):
    pixel = np.zeros((1, 1, 3))
    pixel[0][0] = np.array(rgb)
    pixel = pixel.astype(np.uint8)

    return cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)

def get_color_bounds(pixel, H_margin : list, S_margin: list, V_margin: list):
    H = pixel[0][0][0]
    S = pixel[0][0][1]
    V = pixel[0][0][2]

    lower_bound = np.array([H-H_margin[0], S-S_margin[0], V-V_margin[0]])
    upper_bound = np.array([H+H_margin[1], S+S_margin[1], V+V_margin[1]])

    for bound in [lower_bound, upper_bound]:
        for idx, val in enumerate(bound):
            if val > 255:
                bound[idx] = 255

    return (lower_bound, upper_bound)


def get_signle_ransac(draw_img: np.ndarray, ransac_img: np.ndarray):
    #   input:
    #   the final result on the draw_img
    #   get the ransac processing on the ransac_img

    #   output:
    #   the image with a line on it

    # extract the array of X and Y location of all 'white' pixels in the mask image
    x_coords = []
    y_coords = []

    for row_num in range(ransac_img.shape[0]):
        for col_num in range(ransac_img.shape[1]):
            if ransac_img[row_num][col_num] == 255:
                x_coords.append(col_num)
                y_coords.append(row_num)

    x_coords = np.reshape(x_coords, (-1, 1))
    y_coords = np.reshape(y_coords, (-1, 1))

    ransac_line = linear_model.RANSACRegressor(residual_threshold=10)
    ransac_line.fit(x_coords, y_coords)

    # draw the line on the image
    line_X = np.arange(x_coords.min(), x_coords.max())[:, np.newaxis]
    line_y = ransac_line.predict(line_X)

    cv2.line(draw_img, (line_X[0], line_y[0]), (line_X[-1], line_y[-1]), (0, 0, 255), 1)
    return draw_img


def draw_hough_lines(draw_img_input: np.ndarray, edges_img: np.ndarray, threshold: int, rho_res: int = 1, theha_res = np.pi/180):
    draw_img = np.copy(draw_img_input)
    lines = cv2.HoughLines(edges_img, rho=rho_res, theta=theha_res, threshold=threshold)

    draw_hough_lines_on_img(lines, draw_img)
    return (draw_img, lines)


def draw_hough_lines_on_img(lines: np.ndarray, input_img, color=(0, 0, 255), thickness=1):

    # draw on input_img the lines
    if lines is not None:
        for line_num in range(lines.shape[0]):
            rho = lines[line_num][0][0]
            theta = lines[line_num][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(input_img, (x1, y1), (x2, y2), color, thickness)
    return input_img

def get_common_lines(lines_base: np.ndarray, lines_mask: np.ndarray, threshold_rho: int, threshold_theta: float):

    # calculate the euclidean distance between the parameters of each line in both arrays
    # save only the lines where the difference is smaller than threshold

    new_lines = []
    lines_arr = None

    def inner_loop(rho_1, theta_1):
        for line_num_2 in range(lines_mask.shape[0]):
            rho_2 = lines_mask[line_num_2][0][0]
            theta_2 = lines_mask[line_num_2][0][1]

            theta_1 = np.arctan2(np.sin(theta_1), np.cos(theta_1))
            theta_2 = np.arctan2(np.sin(theta_2), np.cos(theta_2))

            if np.abs(rho_1 - rho_2) <= threshold_rho and np.abs(theta_1 - theta_2) <= threshold_theta:
                new_lines.append([rho_1, theta_1])
                return

    if lines_base is not None and lines_mask is not None:
        for line_num_1 in range(lines_base.shape[0]):
            rho_1 = lines_base[line_num_1][0][0]
            theta_1 = lines_base[line_num_1][0][1]
            inner_loop(rho_1, theta_1)

    if len(new_lines) > 0:
        new_lines_arr =  np.asarray(new_lines)
        lines_arr = np.expand_dims(new_lines_arr, 1)

    return lines_arr


def non_max_suppression_lines(lines: np.ndarray, threshold_rho: int, threshold_theta: float):
    # Find similar lines inside the defined boundaries
    # Take their mean

    new_lines = []
    lines_arr = None
    suppressed_lines = []

    if lines is not None:
        for line_num in range(lines.shape[0]):
            rho = lines[line_num][0][0]
            theta = lines[line_num][0][1]
            theta = np.arctan2(np.sin(theta), np.cos(theta))  # normalize
            met_suppressed_line = False  # is not, create new line

            for suppressed_line in suppressed_lines:
                rho_supp = suppressed_line[0]
                theta_supp = suppressed_line[1]

                if np.abs(rho - rho_supp) <= threshold_rho and np.abs(theta - theta_supp) <= threshold_theta:
                    suppressed_line[0] = (rho_supp + rho)/2  # mean
                    suppressed_line[1] = (theta_supp + theta) / 2  # mean
                    met_suppressed_line = True

            if not met_suppressed_line:
                suppressed_lines.append([rho, theta])

    if len(suppressed_lines) > 0:
        new_lines_arr =  np.asarray(suppressed_lines)
        lines_arr = np.expand_dims(new_lines_arr, 1)
    return lines_arr


def find_intersections(lines: np.ndarray, original_img):
    # find the intersections between the points,
    # which lay inside the boundaries of the image

    draw_img = np.copy(original_img)

    # find intersection point of each line with other 3 lines
    # return only points

    # using the K-Means we segment the lines into vertical and horizontal
    segmented = segment_by_angle_kmeans(lines)
    intersections = segmented_intersections(segmented)


    colors = [(0, 255, 0), (0, 0, 255)]

    for segment, color in zip(segmented, colors):
        segment_arr = np.asarray(segment)
        draw_img = draw_hough_lines_on_img(segment_arr, draw_img, color, thickness=2)

    for intersec in intersections:
        x = intersec[0][0]
        y = intersec[0][1]
        cv2.circle(draw_img, (x, y), 10, (255, 0, 0), thickness=-1)

    return intersections, segmented, draw_img


def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2))

    return intersections


def draw_polygons(intersection_points: list, original_img):
    draw_img = np.copy(original_img)

    # change intersections 3 and 4 places
    intersection_points[2], intersection_points[3] = intersection_points[3], intersection_points[2]

    inter_points = np.asarray(intersection_points)
    # draw the polygon within the intersection points
    draw_img = cv2.fillConvexPoly(img=draw_img, points=inter_points, color=(0, 0, 0))
    return draw_img



def get_segmentation(input_img_loc: str, debug: bool=True, debug_folder: str="output_debug"):
    """This process receives an image, and outputs its segmentation"""
    if debug:
        if not os.path.exists(debug_folder):
            os.mkdir(debug_folder)

    # Main parameters of the process
    # 1. filter color
    # We are interested mainly in the Hue value, which is quiet saturated. Define a margin of +- 10 and filter the original image
    H_margin = [6, 6]
    S_Margin = [40, 100]
    V_margin = [120, 100]

    # 2. Morphological 'open' operation
    kernel = np.ones((3, 3), np.uint8)

    # 3. Canny edge detector
    canny_threshold_1 = 50
    canny_threshold_2 = 150
    canny_aperture_size = 3

    # 4. Hough on original edges
    hough_threshold_normal = 100

    # 5. Hough on masked image edges
    hough_threshold_masked = 20

    # 6. threshold to find common lines
    threshold_rho_common = 3
    threshold_theta_common = 0.2

    # 7. threshold for non max suppression
    threshold_rho_max_sup = 20
    threshold_theta_max_sup = 0.5

    img_cv = cv2.imread(input_img_loc)
    img_mp = mpimg.imread(input_img_loc)

    if debug:
        cv2.imwrite(os.path.join(debug_folder, '1_input_img.jpg'), img_cv)

    img_cv = cv2.resize(img_cv, None, fx=0.2, fy=0.2)
    img_mp = cv2.resize(img_mp, None, fx=0.2, fy=0.2)

    img2_hsv = cv2.cvtColor(img_mp, cv2.COLOR_RGB2HSV)

    # get the yellow pixel in hsv
    # the yellow is approximately [193 165 65]  in RGB
    yellow_pixel = rgb_2_hsv_pixel([193, 165, 65])
    # the yellow is approximately [23 169 169]  in HSV




    (lower_yellow, upper_yellow) = get_color_bounds(yellow_pixel, H_margin, S_Margin, V_margin)

    mask_img = cv2.inRange(img2_hsv, lower_yellow, upper_yellow)

    if debug:
        cv2.imwrite(os.path.join(debug_folder, '2_mask_before_morph.jpg'), mask_img)
    # use morphological 'open'

    mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel)

    if debug:
        cv2.imwrite(os.path.join(debug_folder, '3_mask_after_morph.jpg'), mask_img)



    # Using the RANSAC algorithm to find the lines which resemble the lines of the doors
    # Regular algorithms find only one model using the RANSAC algorithm.
    # We here have to find numerous lines, which will sum to 3 after the non-max suppression.

    # original_img = np.copy(img_cv)
    # ransac_img = get_signle_ransac(original_img, mask_img)
    # cv2.imshow('RANSAC image', ransac_img)
    # cv2.imwrite('output/RANSAC_example.jpg', ransac_img)


    # Using the Hough transform algorithm to find the lines
    # first demonstrate on original image
    original_img = np.copy(img_cv)
    gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=canny_threshold_1, threshold2=canny_threshold_2, apertureSize=canny_aperture_size)

    if debug:
        cv2.imwrite(os.path.join(debug_folder, '4_Edges_original_image.jpg'), edges)

    hough_image, lines_on_normal_image = draw_hough_lines(original_img, edges, threshold=hough_threshold_normal)

    if debug:
        cv2.imwrite(os.path.join(debug_folder, '5_Houghlines_on_original_image.jpg'), hough_image)


    # now use on the image with the mask
    mask_img_copy = np.copy(mask_img)
    edges_mask = cv2.Canny(mask_img_copy, threshold1=canny_threshold_1, threshold2=canny_threshold_2, apertureSize=canny_aperture_size)

    if debug:
        cv2.imwrite(os.path.join(debug_folder, '6_Edges_of_the_mask_image.jpg'), edges_mask)


    # mask_img_2_colored = np.expand_dims(mask_img_copy, 0)
    # mask_img_2_colored = np.concatenate((mask_img_2_colored, mask_img_2_colored, mask_img_2_colored), 0)
    # mask_img_2_colored = np.moveaxis(mask_img_2_colored, 0, -1)

    # original_img = np.copy(img_cv)
    hough_image_2, lines_on_masked_img = draw_hough_lines(original_img, edges_mask, threshold=hough_threshold_masked)
    if debug:
        cv2.imwrite(os.path.join(debug_folder, '7_Houghlines_on_original_image_from_mask.jpg'), hough_image_2)


    # To get better result, use the mask
    # for each line found from original img, check if it includes the mask image
    #   new_mask = cv2.bitwise_and(edges, edges, mask = mask_img_2)

    # get the lines which are similar between the ones extracted from the normal image
    # and the masked image

    common_lines = get_common_lines(lines_on_normal_image, lines_on_masked_img, threshold_rho=threshold_rho_common, threshold_theta=threshold_theta_common)

    # draw those lines on the original image
    filtered_lines_image = np.copy(original_img)
    filtered_lines_image = draw_hough_lines_on_img(common_lines, filtered_lines_image)
    if debug:
        cv2.imwrite(os.path.join(debug_folder, '8_Merged_lines_on_original_image.jpg'), filtered_lines_image)

    if common_lines is not None:
        print("Commong lines detected: " + str(common_lines.shape[0]))


    # Use the non-max suppression to unite common lines
    non_max_common_lines = non_max_suppression_lines(common_lines, threshold_rho=threshold_rho_max_sup, threshold_theta=threshold_theta_max_sup)

    # draw those lines on the original image
    filtered_suppressed_lines_image = np.copy(original_img)
    filtered_suppressed_lines_image = draw_hough_lines_on_img(non_max_common_lines, filtered_suppressed_lines_image)
    if debug:
        cv2.imwrite(os.path.join(debug_folder, '9_Suppressed_lines_on_original_image.jpg'), filtered_suppressed_lines_image)

    if common_lines is not None:
        print("Suppressed lines detected: " + str(non_max_common_lines.shape[0]))


    # To continue, we NEED to have EXACTLY 4 lines.

    # Find 4 intersection points
    (intersection_points, segmented_lines, intersections_image) = find_intersections(non_max_common_lines, original_img)

    if debug:
        cv2.imwrite(os.path.join(debug_folder, '10_Intersections_image.jpg'), intersections_image)

    if len(segmented_lines[0]) == 2 and len(segmented_lines[1]) == 2:
        # draw the polygon inside
        polygon_image = draw_polygons(intersection_points, original_img)
        if debug:
            cv2.imwrite(os.path.join(debug_folder, '11_Polygon.jpg'), polygon_image)
    else:
        polygon_image = None

    return polygon_image
