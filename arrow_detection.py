
import cv2
import numpy as np
import general_pipeline as gp

import color_util as cu

def show(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)

# arrowbases = array of positions of arrow starts
# arrowheads = array of positions of arrow_ends

arrowbase_hue_range = cu.grn_hue_range
arrowhead_hue_range = cu.red_hue_range

def base_mask(hsv_img):
    return cu.hue_mask(arrowbase_hue_range, 50, hsv_img)

def head_mask(hsv_img):
    return cu.hue_mask(arrowhead_hue_range, 50, hsv_img)

# TODO: Function that takes an img and returns the centroids of the heads
# TODO: Function that takes an array of base centroids, an array of head centroids,
#   and the image, then returns an array of [base, head] pairs

image = cv2.imread("images/tiny_arrowtest.png") # DEBUG
hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # DEBUG
hues = hsv_img[:,:,0] # DEBUG
print(hues) # DEBUG
show("hues", hues) # DEBUG
show("bases", base_mask(hsv_img)) # DEBUG
show("heads", head_mask(hsv_img)) # DEBUG
cv2.waitKey(0)
