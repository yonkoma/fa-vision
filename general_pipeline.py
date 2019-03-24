
import numpy as np
import cv2
from math import log

LOG_255 = log(255)

# Blurs a gray image.
def blur (img, blurSize):
    blurred = cv2.GaussianBlur(img, (blurSize, blurSize), 0)
    return blurred

# Exponential transform of image.
# maps 0 to 0 and 255 to 255
def exp (img):
    normalizedImage = (img/255)
    exponentsImage = normalizedImage * LOG_255
    result = np.clip(np.exp(exponentsImage),0,255).astype(np.uint8)
    return result

# Thresholds a gray image.
def threshold (img, offset=0):
    otsu_thresh, ret2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret1, thresholded = cv2.threshold(img, otsu_thresh + offset, 255, cv2.THRESH_BINARY)
    return thresholded

# Given a binary image of shapes that have holes,
#   returns a binary image 
#   with only the holes of the shapes in white.
def only_holes(img):
    result = img.copy()

    cv2.floodFill(result, None, (0, 0), 255)
    result = cv2.bitwise_not(result)

    return result

# Fills all holes of every blob
#   in a binary image.
# For more info see the following link:
# https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
def fill_blobs(img):
    return cv2.bitwise_or(only_holes(img), img)

# Morphological close operation.
# Uses a circular ellipse kernel.
def close(img, size, iterations=1):
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
            iterations=iterations)

# Morphological open operation.
# Uses a circular ellipse kernel.
def open(img, size, iterations=1):
    return cv2.morphologyEx(img, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
            iterations=iterations)

# Counts the amount of connected components in a binary image.
# Subtracts one due to the whole image being
#   counted as a connected component.
def count_connected_components(img):
    return cv2.connectedComponents(img)[0] - 1

# Takes in an image and some stats.
# Returns the image with thick red rectangles around the spaces
#   bounded by the stats.
def rectanglify(img, stats, grayscale=False):
    result = img.copy()

    for [xleft, ytop, width, height] in stats:
        xright = xleft+width
        ybot = ytop+height
        color = 255 if grayscale else (0,0,255)
        cv2.rectangle(result, (xleft, ytop), (xright, ybot), color, 6)

    return result

def resize(img, height):
    """
    Resize the image to fit the given height while keeping proportions.
    Returns the factor by which the image was scaled in addition to the new image
    """
    size = (round(img.shape[1]*height/img.shape[0]), height)
    return cv2.resize(img, size), height/img.shape[0]