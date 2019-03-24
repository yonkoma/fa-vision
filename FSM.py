import cv2
import numpy as np
import math
import general_pipeline as gpipe

def log_trans(img, fact):
    if fact == 0:
        return img
    img = img / 255 * fact + 1
    img = np.log(img)
    img = 255 * img / np.log(1 + fact)
    img = img.astype(np.uint8)
    return img

def __get_rect_roi__(image, rect):
    center, size, theta = rect
    size = tuple(map(int, size))
    theta *= 3.14159 / 180  # convert to rad
    v_x = (math.cos(theta), math.sin(theta))
    v_y = (-math.sin(theta), math.cos(theta))
    s_x = center[0] - v_x[0] * ((size[0] - 1) / 2) - v_y[0] * ((size[1] - 1) / 2)
    s_y = center[1] - v_x[1] * ((size[0] - 1) / 2) - v_y[1] * ((size[1] - 1) / 2)
    mapping = np.array([[v_x[0], v_y[0], s_x], [v_x[1], v_y[1], s_y]])
    return cv2.warpAffine(image, mapping, size, flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)

def __flood_fill_corners__(image, color):
    h, w = image.shape[:2]
    cv2.floodFill(image, None, (0, 0), color)
    cv2.floodFill(image, None, (w - 1, 0), color)
    cv2.floodFill(image, None, (0, h - 1), color)
    cv2.floodFill(image, None, (w - 1, h - 1), color)


# Image Preprocessing
print("Reading File...", end="")
image = cv2.imread("images/paper2.jpg")
filtered = cv2.bilateralFilter(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 11, 17, 17)
log = log_trans(filtered, 20)
print("\rPreprocessing Image Data...", end="")
otsu_thresh, thresh = cv2.threshold(log, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
__flood_fill_corners__(thresh, 0)
img, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    if 2000 < cv2.contourArea(c):
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))
        subimage = __get_rect_roi__(image, rect)
        cv2.imshow("thing", subimage)
        cv2.waitKey(0)
        cv2.drawContours(filtered, [box], 0, (255, 0, 0), 10)


cv2.imshow("testc", filtered)
thresh = cv2.bitwise_not(thresh)
cv2.imshow("test", thresh)
cv2.waitKey(0)




