import cv2
import numpy as np

def __flood_fill_corners__(image, color):
    h, w = image.shape[:2]
    cv2.floodFill(image, None, (0, 0), color)
    cv2.floodFill(image, None, (w - 1, 0), color)
    cv2.floodFill(image, None, (0, h - 1), color)
    cv2.floodFill(image, None, (w - 1, h - 1), color)


# Image Preprocessing
print("Reading File...", end="")
image = cv2.imread("IMG_2622.jpg")
print("\rPreprocessing Image Data...", end="")
filtered = cv2.bilateralFilter(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 11, 17, 17)



otsu_thresh, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
otsu_thresh, real_thresh = cv2.threshold(filtered, otsu_thresh + 25, 255, cv2.THRESH_BINARY)



kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
closing = cv2.morphologyEx(real_thresh, cv2.MORPH_OPEN, kernel, iterations=12)

#__flood_fill_corners__(real_thresh, 0)


cv2.imshow("test", real_thresh)
cv2.waitKey(0)








