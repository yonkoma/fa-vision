import cv2
import numpy as np
import general_pipeline as gpipe

def log_trans(img, fact):
    if fact == 0:
        return img
    img = img / 255 * fact + 1
    img = np.log(img)
    img = 255 * img / np.log(1 + fact)
    img = img.astype(np.uint8)
    return img

def __flood_fill_corners__(image, color):
    h, w = image.shape[:2]
    cv2.floodFill(image, None, (0, 0), color)
    cv2.floodFill(image, None, (w - 1, 0), color)
    cv2.floodFill(image, None, (0, h - 1), color)
    cv2.floodFill(image, None, (w - 1, h - 1), color)


# Image Preprocessing
print("Reading File...", end="")
image = cv2.imread("images/paper1.jpg")
filtered = cv2.bilateralFilter(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 11, 17, 17)
log = log_trans(filtered, 20)
print("\rPreprocessing Image Data...", end="")
otsu_thresh, thresh = cv2.threshold(log, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
otsu_thresh, real_thresh = cv2.threshold(filtered, otsu_thresh + 25, 255, cv2.THRESH_BINARY)



kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
closing = cv2.morphologyEx(real_thresh, cv2.MORPH_OPEN, kernel, iterations=12)

__flood_fill_corners__(thresh, 0)

thresh = cv2.bitwise_not(thresh)
cv2.namedWindow("test", cv2.WINDOW_NORMAL)

cv2.imshow("test", thresh)
cv2.waitKey(0)








