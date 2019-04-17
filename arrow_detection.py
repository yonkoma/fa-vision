
import cv2
import numpy as np
import general_pipeline as gp
# import sys # DEBUG

import color_util as cu

def show(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)

# arrowbases = list of arrow_base centroids
arrowbases = []
# arrowheads = list of arrow_head centroids
arrowheads = []

arrowbase_hue_range = cu.grn_hue_range
arrowhead_hue_range = cu.red_hue_range

# Returns a mask for the bases of the arrows
def base_mask(img):
    blurred = gp.blur(img, 11)
    hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cu.color_mask(arrowbase_hue_range, 80, hsv_img)
    open_mask = gp.open(mask, 3)
    return open_mask

# Returns a mask for the heads of the arrows
def head_mask(img):
    blurred = gp.blur(img, 11)
    hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cu.color_mask(arrowhead_hue_range, 80, hsv_img)
    open_mask = gp.open(mask, 3)
    return open_mask

# Takes an hsv image, its binary image, and a list of state centers,
#   then returns a version of the image with only the arrows & their labels
def arrow_mask(hsv_img, bin_img, state_centers):
    base_and_head_mask = cv2.bitwise_and(base_mask(hsv_img), head_mask(hsv_img))

    # Remove bases and heads, disconnecting paths from states
    result = cv2.bitwise_and(bin_img, cv2.bitwise_not(base_and_head_mask))

    # Fill each state blob
    result = gp.fill_blobs(result)

    # Remove each state blob
    for [x, y] in state_centers:
        cv2.floodFill(result, None, (int(x), int(y)), 0)

    open_result = gp.open(result, 3)
    return open_result

# TODO: Function that takes a base mask, a head mask, and a state mask,
#   then returns a list an array of [base, head]

# TODO: Function that takes an array of base centroids, an array of head centroids,
#   and the image, then returns an array of [base, head] pairs

# image = cv2.imread(sys.argv[1]) # DEBUG
# show("img", image) # DEBUG
# hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # DEBUG
# hues = hsv_img[:,:,0] # DEBUG
# show("hues", hues) # DEBUG
# show("bases", base_mask(hsv_img)) # DEBUG
# show("heads", head_mask(hsv_img)) # DEBUG
# cv2.waitKey(0) # DEBUG
