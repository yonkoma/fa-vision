import cv2
import numpy as np
import general_pipeline as gpipe

def show(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)

# Image Preprocessing
print("Reading File...", end="")
image = cv2.imread("images/paper2.jpg")

image_size = max(image.shape[0], image.shape[1])

MIN_STATE_AREA = int(((1/2) * image_size))
print(MIN_STATE_AREA)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

filtered = cv2.bilateralFilter(gray, 11, 17, 17)
log = gpipe.log(filtered, 20)

show("log", log)

print("\rPreprocessing Image Data...", end="")
otsu_thresh, thresh = cv2.threshold(log, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

show("thresh", thresh)

floodfilled = gpipe.flood_fill_corner(thresh, 0)

contours, hierarchy = cv2.findContours(floodfilled.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    if MIN_STATE_AREA < cv2.contourArea(c):
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))
        subimage = gpipe.get_rect(image, rect)
        show("thing", subimage)
        cv2.waitKey(0)
        cv2.drawContours(filtered, [box], 0, (255, 0, 0), 10)


show("testc", filtered)
thresh = cv2.bitwise_not(thresh)
show("test", thresh)
cv2.waitKey(0)




