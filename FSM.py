import cv2
import numpy as np
import general_pipeline as gpipe
import arrow_detection as adetect
import argparse

HALF_SHIFT = 1

def show(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)

def removeGlare(img, ksize, area_thresh):
    sobel_x = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=ksize)
    
    _, thresh_x = cv2.threshold(sobel_x, 250, 255, cv2.THRESH_BINARY)
    _, thresh_y = cv2.threshold(sobel_y, 250, 255, cv2.THRESH_BINARY)

    log = gpipe.log(img, 20)
    otsu_thresh, thresh = cv2.threshold(cv2.bitwise_not(log), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    shift = HALF_SHIFT*2
    comp = cv2.bitwise_or(thresh_x[:-shift,shift:], thresh_y[shift:,:-shift])

    comp = cv2.bitwise_or(comp, thresh[HALF_SHIFT:-HALF_SHIFT, HALF_SHIFT:-HALF_SHIFT])


    count, markers, stats, centroids = cv2.connectedComponentsWithStats(comp, connectivity=4)
    for i in range(1, count):
        if stats[i][cv2.CC_STAT_AREA] < area_thresh:
            comp[markers == i] = 0
    
    return comp

def main(args):
    # Image Preprocessing
    print("Reading Image... ", end="")
    image = cv2.imread(args.image)
    image, _ = gpipe.resize(image, 800)

    image_size = max(image.shape[0], image.shape[1])

    MIN_STATE_AREA = int(((1/2) * image_size))
    print("Size: " + str(MIN_STATE_AREA))
    print("Preprocessing Image Data...")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    
    if args.glare:
        anti_glare = removeGlare(filtered, 5, 100)
        thresh = cv2.bitwise_not(anti_glare)
    else:
        otsu_thresh, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        basemask = adetect.base_mask(image)
        headmask = adetect.head_mask(image)

        base_or_head_mask = cv2.bitwise_or(basemask, headmask)

        thresh = cv2.bitwise_and(cv2.bitwise_not(thresh), cv2.bitwise_not(base_or_head_mask))
        thresh = cv2.bitwise_not(thresh)
       
    show("thresh", thresh)
    floodfilled = gpipe.flood_fill_corner(thresh, 0)

    # lazy version check
    lazyver = cv2.__version__[0]
    if lazyver == '3':
        ret1, contours, hierarchy = cv2.findContours(floodfilled.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    elif lazyver == '4':
        contours, hierarchy = cv2.findContours(floodfilled.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        raise Exception("Bad cv2 version check")

    rects = []
    # Extract each state from the image
    for c in contours:
        # If the contour is the right size
        if MIN_STATE_AREA < cv2.contourArea(c):
            rect = cv2.minAreaRect(c)
            rects.append(rect)
            box = np.int0(cv2.boxPoints(rect))
            subimage = gpipe.get_rect(image, rect)
            show("thing", subimage)
            cv2.waitKey(0)
            cv2.drawContours(filtered, [box], 0, (255, 0, 0), 10)

    centers = [center for center, size, theta in rects]

    cv2.waitKey(0)

    base_to_heads = adetect.base_to_head_centroids(image, cv2.bitwise_not(thresh), centers)

    if args.debug:
        for [[x1,y1], [x2,y2]] in base_to_heads:
            cv2.circle(image, (x1,y1), 10, (0, 255, 0), thickness=5)
            cv2.circle(image, (x2,y2), 10, (0, 0, 255), thickness=5)
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), thickness=5)

    arrow_mask = adetect.arrow_mask(image, cv2.bitwise_not(thresh), centers)
    show("base or arrow mask", cv2.bitwise_or(arrow_mask, adetect.base_mask(image)))
    show("head or arrow mask", cv2.bitwise_or(arrow_mask, adetect.head_mask(image)))
    show("image", image)
    show("testc", filtered)
    thresh = cv2.bitwise_not(thresh)
    show("test", thresh)
    cv2.waitKey(0)

parser = argparse.ArgumentParser(description='Recognize DFAs.')
parser.add_argument('image', metavar='IMAGE', type=str, help='the image of the DFA to process.')
parser.add_argument('-g', '--glare', action='store_true', help='do anti-glare pre-processing.')
parser.add_argument('-d', '--debug', action='store_true', help='show debug lines.')
args = parser.parse_args()
main(args)




