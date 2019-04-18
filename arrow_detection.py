
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
    mask = cu.color_mask(arrowbase_hue_range, 40, hsv_img)
    open_mask = gp.open(mask, 3)
    return open_mask

# Returns a mask for the heads of the arrows
def head_mask(img):
    blurred = gp.blur(img, 11)
    hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cu.color_mask(arrowhead_hue_range, 80, hsv_img)
    open_mask = gp.open(mask, 3)
    return open_mask

def arrow_or_label_mask(img, bin_img, state_centers):
    bases = base_mask(img)
    heads = head_mask(img)

    base_or_head_mask = cv2.bitwise_or(bases, heads)

    # Remove bases and heads, disconnecting paths from states
    result = cv2.bitwise_and(bin_img, cv2.bitwise_not(base_or_head_mask))

    # Fill each state blob
    result = gp.fill_blobs(result)

    # Remove each state blob
    for [x, y] in state_centers:
        cv2.floodFill(result, None, (int(x), int(y)), 0)

    return result

# Takes an hsv image, its binary image, and a list of state centers,
#   then returns a version of the image with only the arrows & their labels
def arrow_mask(img, bin_img, state_centers):
    bases = base_mask(img)

    base_centroids = cv2.connectedComponentsWithStats(bases)[3][1:]
    base_centroids = [[int(x), int(y)] for [x, y] in base_centroids]

    # Get the arrows and the labels
    result = arrow_or_label_mask(img, bin_img, state_centers)

    # Add bases (will be removed later)
    result = cv2.bitwise_or(result, bases)

    # Color the arrows in a specific hue
    for [x, y] in base_centroids:
        cv2.floodFill(result, None, (int(x), int(y)), 1)

    # Remove bases
    result = cv2.bitwise_and(result, cv2.bitwise_not(bases))

    result[result != 1] = 0
    result *= 255

    return result

def label_mask(img, bin_img, state_centers):
    bases = base_mask(img)

    base_centroids = cv2.connectedComponentsWithStats(bases)[3][1:]
    base_centroids = [[int(x), int(y)] for [x, y] in base_centroids]

    # Get the arrows and the labels
    result = arrow_or_label_mask(img, bin_img, state_centers)

    # Add bases (will be removed later)
    result = cv2.bitwise_or(result, bases)

    # Remove the arrows and bases
    for [x, y] in base_centroids:
        cv2.floodFill(result, None, (int(x), int(y)), 0)

    return result

def base_to_head_centroids(img, bin_img, state_centers):
    bases = base_mask(img)
    heads = head_mask(img)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    arrows = arrow_mask(img, bin_img, state_centers)

    base_centroids = cv2.connectedComponentsWithStats(bases)[3][1:]
    base_centroids = [[int(x), int(y)] for [x, y] in base_centroids]

    head_centroids = cv2.connectedComponentsWithStats(heads)[3][1:]
    head_centroids = [[int(x), int(y)] for [x, y] in head_centroids]

    base_head_arrow_mask = cv2.bitwise_or(bases, cv2.bitwise_or(heads, arrows))
    base_head_arrow_mask = gp.dilate(base_head_arrow_mask, 5)

    # show("bham", base_head_arrow_mask)
    # cv2.waitKey(0)

    result = []
    for i, base_centroid in enumerate(base_centroids):
        cv2.floodFill(base_head_arrow_mask, None, (base_centroid[0], base_centroid[1]), i + 1)

    for head_centroid in head_centroids:
        label = base_head_arrow_mask[head_centroid[1], head_centroid[0]]
        base_centroid = base_centroids[label - 1]
        result.append([base_centroid, head_centroid])

    return result


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
