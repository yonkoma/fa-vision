
import numpy as np
import cv2
import math

LOG_255 = math.log(255)

"""
Blurs a gray image.
"""
def blur (img, blurSize):
    blurred = cv2.GaussianBlur(img, (blurSize, blurSize), 0)
    return blurred

"""
Exponential transform of image.
Maps 0 to 0 and 255 to 255
"""
def exp (img):
    result = (img/255)
    result = result * LOG_255
    result = np.clip(np.exp(exponentsImage),0,255).astype(np.uint8)
    return result

"""
Logarithmic transform of an image. 
"""
def log(img, fact):
    img = img.copy()
    if fact == 0:
        return img.copy()
    img = img * (fact / 255) + 1
    img = np.log(img)
    img = 255 * img / np.log(1 + fact)
    img = img.astype(np.uint8)
    return img

"""
Thresholds a gray image.
"""
def threshold (img, offset=0):
    otsu_thresh, ret2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret1, thresholded = cv2.threshold(img, otsu_thresh + offset, 255, cv2.THRESH_BINARY)
    return thresholded

"""
Given a binary image of shapes that have holes,
  returns a binary image 
  with only the holes of the shapes in white.
"""
def only_holes(img):
    result = img.copy()

    cv2.floodFill(result, None, (0, 0), 255)
    result = cv2.bitwise_not(result)

    return result

"""
Given a binary image, floodfills in a color
  from the top left corner.
"""
def flood_fill_corner(img, color):
    img = img.copy()
    cv2.floodFill(img, None, (0, 0), color)
    return img

"""
Fills all holes of every blob
  in a binary image.
For more info see the following link:
https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
"""
def fill_blobs(img):
    return cv2.bitwise_or(only_holes(img), img)

"""
Gets each centroid of each blob in an image
"""
def centroids(img):
    return cv2.connectedComponentsWithStats(img)[3][1:]

"""
Gets each centroid of each blob in an image, floored to an int
"""
def int_centroids(img):
    return [[int(x), int(y)] for [x, y] in centroids(img)]

"""
Morphological close operation.
Uses a circular ellipse kernel.
"""
def close(img, size, iterations=1):
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
            iterations=iterations)

"""
Morphological open operation.
Uses a circular ellipse kernel.
"""
def open(img, size, iterations=1):
    return cv2.morphologyEx(img, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
            iterations=iterations)

"""
Morphological dilate operation.
Uses a circular ellipse kernel.
"""
def dilate(img, size, iterations=1):
    return cv2.dilate(img,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
        iterations)

"""
Removes components of a binary image that have below a certain area
"""
def removeSmallComponents(bin_img, area_size):
    count, markers, stats, centroids = cv2.connectedComponentsWithStats(bin_img, connectivity=4)
    for i in range(1, count):
        if stats[i][cv2.CC_STAT_AREA] < area_size:
            bin_img[markers == i] = 0

    return bin_img

"""
Counts the amount of connected components in a binary image.
Subtracts one due to the whole image being
  counted as a connected component.
"""
def count_connected_components(img):
    return cv2.connectedComponents(img)[0] - 1

"""
Takes in an image and some stats.
Returns the image with thick red rectangles around the spaces
  bounded by the stats.
"""
def rectanglify(img, stats, grayscale=False):
    result = img.copy()

    for [xleft, ytop, width, height] in stats:
        xright = xleft+width
        ybot = ytop+height
        color = 255 if grayscale else (0,0,255)
        cv2.rectangle(result, (xleft, ytop), (xright, ybot), color, 6)

    return result

"""
Resize the image to fit the given height while keeping proportions.
Returns the new image followed by the scaling factor used.
"""
def resize(img, height):
    size = (round(img.shape[1]*height/img.shape[0]), height)
    return cv2.resize(img, size), height/img.shape[0]

"""
Extracts a rectangle from an image.
Rotates the rectangle so it fits in the upright window.
"""
def get_rect(image, rect):
    center, size, theta = rect
    size = tuple(map(int, size))
    theta *= 3.14159 / 180  # convert to rad
    v_x = (math.cos(theta), math.sin(theta))
    v_y = (-math.sin(theta), math.cos(theta))
    s_x = center[0] - v_x[0] * ((size[0] - 1) / 2) - v_y[0] * ((size[1] - 1) / 2)
    s_y = center[1] - v_x[1] * ((size[0] - 1) / 2) - v_y[1] * ((size[1] - 1) / 2)
    mapping = np.array([[v_x[0], v_y[0], s_x], [v_x[1], v_y[1], s_y]])
    return cv2.warpAffine(image, mapping, size, flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)

"""
Uses sobel gradients to find lines in image. Resultsing binary image is merged with supplied thresholded image
for robustness.
"""
def removeGlare(img, thresh, ksize, area_thresh, half_shift=1):
    sobel_x = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=ksize)

    _, thresh_x = cv2.threshold(sobel_x, 240, 255, cv2.THRESH_BINARY)
    _, thresh_y = cv2.threshold(sobel_y, 240, 255, cv2.THRESH_BINARY)

    shift = half_shift*2
    comp = np.zeros(img.shape, dtype=img.dtype)
    comp[half_shift:-half_shift, half_shift:-half_shift] = cv2.bitwise_or(thresh_x[:-shift,shift:], thresh_y[shift:,:-shift])
    comp = cv2.bitwise_or(comp, thresh)
    comp = removeSmallComponents(comp, area_thresh)

    return comp

"""
Finds the otsu threshold and thresholds the image after making an adjustment to it.
"""
def adjustedOtsu(img, adjust=0):
    otsu_thresh, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresh = cv2.threshold(img, otsu_thresh + adjust, 255, cv2.THRESH_BINARY)
    return thresh

"""
Makes a list of indices to cut in order to split a range of `size` into `cut` sections.
"""
def getRoughCuts(size, cuts):
    cut_list = [0]
    location = 0
    cut_count = 0
    while cut_count < size % cuts:
        location += size // cuts + 1
        cut_list.append(location)
        cut_count += 1
    while cut_count < cuts:
        location += size // cuts
        cut_list.append(location)
        cut_count += 1
    
    return cut_list
    
"""
Performs otsu thresholding seperately on sections of the image.
"""
def sectionedOtsu(img, section_count, adjust=0):
    height_cuts = getRoughCuts(img.shape[0], section_count)
    width_cuts = getRoughCuts(img.shape[1], section_count)
    
    for row in range(section_count):
        for col in range(section_count):
            section_slice = (slice(height_cuts[row], height_cuts[row + 1]),slice(width_cuts[col], width_cuts[col + 1]))
            img[section_slice] = adjustedOtsu(img[section_slice], adjust)
    
    return img