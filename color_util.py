
import cv2
import numpy as np

# These are pixel representatives of
#   the bottom and top of hue ranges
#   for each color.
# The bottom green is a really yellow pixel,
#   but for some reason this value worked better
grn_pxl_range = (np.uint8([[[  0, 255, 234]]]), np.uint8([[[230, 255,   0]]]))
red_pxl_range = (np.uint8([[[  0,  81, 255]]]), np.uint8([[[ 68,   0, 255]]]))

# returns the hue of a pixel
def hue_from_pxl(pxl):
    return cv2.cvtColor(pxl, cv2.COLOR_BGR2HSV)[0,0,0]

grn_hue_range = hue_from_pxl(grn_pxl_range[0], grn_pxl_range[1])
red_hue_range = hue_from_pxl(red_pxl_range[0], red_pxl_range[1])

# Returns the distance between two hues
#   which have a wraparound value.
# e.g. hue_dist(2, 170, 179) = 12
#   because you can just increase 170 by 12 and reach 181
#   which wraps around to 2.
def hue_dist(hue1, hue2, wraparound):
    dist = abs(int(hue1) - int(hue2))

    if dist > wraparound // 2:
        return wraparound - dist

    return np.uint8(dist)

# Runs the hue_dist algorithm between numpy arrays.
# Much faster than doing a for loop.
def numpy_hue_dist(hue_arr1, hue_arr2, wraparound):
    hue_arr1_cpy = np.copy(hue_arr1).astype(int)
    hue_arr2_cpy = np.copy(hue_arr2).astype(int)

    dists = np.absolute(hue_arr1_cpy - hue_arr2_cpy)
    dists[dists > wraparound//2] *= -1
    dists[dists < 0] += wraparound

    return dists

# A hue is "between" two hues
#   if the sum of its distance to each hue
#   is equal to the distance between the two hues.
# Only works if the range between the two hues is less than
#   half of the full range.
#   Otherwise there is not clear answer.
def between_hues(hue, bottom_hue, top_hue, wraparound):
    inbetween_dist = hue_dist(bottom_hue, top_hue, wraparound)
    dist_sum = hue_dist(hue, bottom_hue, wraparound) + hue_dist(hue, top_hue, wraparound)
    return dist_sum == inbetween_dist

# Runs the between_hues algorithm between
#   a numpy arr of hues and a bottom and top hue value
#   returning a boolean array.
def numpy_between_hues(hue_arr1, bottom_hue, top_hue, wraparound):
    inbetween_dist = hue_dist(bottom_hue, top_hue, wraparound)
    dist_sum = (
        numpy_hue_dist(hue_arr1, bottom_hue, wraparound)
        + numpy_hue_dist(hue_arr1, top_hue, wraparound)
    )
    return dist_sum == inbetween_dist
