import cv2
import numpy as np
import general_pipeline as gpipe
import arrow_detection as adetect
import argparse

def show(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)

def main(args):
    # Image Preprocessing
    print("Reading Image... ", end="")
    image = cv2.imread(args.image)

    # Resize image
    image, _ = gpipe.resize(image, 800)

    image_size = max(image.shape[0], image.shape[1])

    # Get minimum state area
    MIN_STATE_AREA = int(((1/2) * image_size))
    print("Preprocessing Image Data...")

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Reduce image noise, keep edges if we'll need them
    blurred = gpipe.blur(gray, 5) if not args.glare else cv2.bilateralFilter(gray, 11, 17, 17)

    # Be a little more demanding than otsu threshold
    thresh = gpipe.adjustedOtsu(blurred, -10)
    # Threshold needs to be inverse so that lines are white on black.
    thresh = cv2.bitwise_not(thresh)
    
    # If glare removal is enabled
    if args.glare:
        thresh = gpipe.removeGlare(blurred, thresh, 5, 200)

    show("thresh", thresh)

    bases = adetect.base_mask(image)
    heads = adetect.head_mask(image)

    base_or_head_mask = cv2.bitwise_or(bases, heads)

    # Remove bases and heads from image to disconnect arrows from states
    thresh = cv2.bitwise_and(thresh, cv2.bitwise_not(base_or_head_mask))

    gpipe.removeSmallComponents(thresh, 10)
       
#    show("thresh", thresh)

    # Floodfill to obtain only the insides of  the state machines (without arrows)
    floodfilled = gpipe.flood_fill_corner(thresh, 255)
    floodfilled = cv2.bitwise_not(floodfilled)

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

            if args.debug:
                # Show the state
                show("thing", subimage)
                cv2.waitKey(0)

            # Draw the contour
            cv2.drawContours(blurred, [box], 0, (255, 0, 0), 10)

    centers = [center for center, size, theta in rects]

    show("testc", blurred)
    show("test", thresh)

    if args.debug:
        show("testc", blurred)
        show("test", thresh)
        show("base mask", bases)
        show("head mask", heads)
        show("arrow mask", adetect.arrow_mask(image, thresh, centers))
        show("label mask", adetect.label_mask(image, thresh, centers))

        labeldims_bases_heads = adetect.label_dim_base_to_head_centroids(image, thresh, centers)
        labeldims = [labeldim for labeldim, base, head in labeldims_bases_heads]

        # Draw lines from each arrow base to each arrow head,
        # and highlight each arrow base and each arrow head
        centroid_base_to_heads = adetect.base_to_arrow_to_head_centroids(image, thresh, centers)
        for i, [[x1,y1], [xc, yc], [x2,y2]] in enumerate(centroid_base_to_heads):
            xc = int(xc)
            yc = int(yc)
            cv2.circle(image, (x1,y1), 10, (0, 255, 0), thickness=5)
            cv2.circle(image, (x2,y2), 10, (0, 0, 255), thickness=5)
            cv2.line(image, (x1, y1), (xc, yc), (255, 0, 0), thickness=5)

            cv2.rectangle(image,
                (labeldims[i][1][0], labeldims[i][0][0]),
                (labeldims[i][1][1], labeldims[i][0][1]),
                (255,0,0), 2)

            show("image", image)
            cv2.waitKey(0)
            cv2.line(image, (xc, yc), (x2, y2), (255, 0, 0), thickness=5)
            show("image", image)
            cv2.waitKey(0)
    show("image", image)

    cv2.waitKey(0)

parser = argparse.ArgumentParser(description='Recognize DFAs.')
parser.add_argument('image', metavar='IMAGE', type=str, help='the image of the DFA to process.')
parser.add_argument('-g', '--glare', action='store_true', help='do anti-glare pre-processing.')
parser.add_argument('-d', '--debug', action='store_true', help='show debug lines.')
args = parser.parse_args()
main(args)




