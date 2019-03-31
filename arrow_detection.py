
import cv2
import numpy as np

import color_util as cu

# arrowbases = array of positions of arrow starts
# arrowheads = array of positions of arrow_ends

arrowbase_hue_range = cu.grn_hue_range
arrowhead_hue_range = cu.red_hue_range

# TODO: Function that takes an img and returns the centroids of the bases
# TODO: Function that takes an img and returns the centroids of the heads
# TODO: Function that takes an array of base centroids, an array of head centroids,
#   and the image, then returns an array of [base, head] pairs
