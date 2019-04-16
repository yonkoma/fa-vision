import cv2
import numpy as np
import math

class ArrowDetector:

    def __init__(self):
        self.speck_filter = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], np.uint8)
        self.cap_filter = np.array([
            [0,2,2,2,2,0],
            [0,0,1,1,0,0],
            [0,0,0,0,0,0]
        ], np.uint8)

    @staticmethod
    def hit_or_miss(image, kernel):
        kernel1 = kernel.copy()
        kernel1[kernel == 2] = 0
        kernel2 = kernel.copy()
        kernel2[kernel > 0] = 0
        kernel2[kernel == 0] = 1
        image_comp = cv2.bitwise_not(image)
        hom      = cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel1)
        hom_comp = cv2.morphologyEx(image_comp, cv2.MORPH_ERODE, kernel2)
        return cv2.bitwise_and(hom, hom_comp)

    def __highlight_caps__(self, image):
        # Filter Standalone Pixels
        image = cv2.bitwise_and(image, cv2.bitwise_not(self.hit_or_miss(image, self.speck_filter)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        image = cv2.dilate(image, kernel, iterations=1)
        cv2.imshow("process", image)
        cv2.imwrite("skeleton.png", image)
        # Highlight Line Caps
        a = self.hit_or_miss(image, self.cap_filter)
        b = self.hit_or_miss(image, np.rot90(self.cap_filter, 1))
        c = self.hit_or_miss(image, np.rot90(self.cap_filter, 2))
        d = self.hit_or_miss(image, np.rot90(self.cap_filter, 3))
        x = cv2.bitwise_or(a,b)
        y = cv2.bitwise_or(c,d)
        return cv2.dilate(cv2.bitwise_or(x,y), kernel, iterations=5)

    @staticmethod
    def __select_far_point__(image, point):
        i, contours, h = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        start_point, max_dist = None, 0
        cv2.circle(image, (int(point[0]), int(point[1])), 10, (255, 255, 255), thickness=5)
        for c in contours:
            center = cv2.minAreaRect(c)[0]
            distance = math.sqrt(math.pow(center[0] - point[0], 2) + math.pow(center[1] - point[1], 2))
            if distance > max_dist:
                max_dist = distance
                start_point = center
        return start_point


    @staticmethod
    def __get_skeleton__(image):
        skeleton = np.zeros(image.shape, np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            eroded = cv2.erode(image, kernel)
            temp = cv2.dilate(eroded, kernel)
            temp = cv2.subtract(image, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            image, eroded = eroded, image
            if cv2.countNonZero(image) == 0:
                return skeleton

    def __parse_arrow__(self, image, box, padding = 10):
        x1, x2 = min(box[:,0]) - padding, max(box[:,0]) + padding
        y1, y2 = min(box[:,1]) - padding, max(box[:,1]) + padding
        sub_image = image[y1:y2, x1:x2]
        cv2.imshow("process", sub_image)
        cv2.imwrite("image.png", sub_image)
        cv2.waitKey(0)
        filtered = cv2.morphologyEx(sub_image, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=4)
        cv2.imshow("process", filtered)
        cv2.imwrite("cleaned.png", filtered)
        cv2.waitKey(0)
        i, contours, h = cv2.findContours(cv2.bitwise_not(filtered), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        end_point, max_size, box = None, 0, None
        for c in contours:
            if cv2.contourArea(c) > max_size:
                max_size = cv2.contourArea(c)
                end_point = cv2.minAreaRect(c)[0]
        skeleton = self.__get_skeleton__(cv2.bitwise_not(sub_image))
        cv2.imshow("process", skeleton)
        cv2.waitKey(0)
        cv2.imshow("process", self.__highlight_caps__(skeleton))
        cv2.imwrite("caps.png", self.__highlight_caps__(skeleton))
        cv2.waitKey(0)
        start_point = self.__select_far_point__(self.__highlight_caps__(skeleton), end_point)
        start_point = (int(start_point[0] + x1), int(start_point[1] + y1))
        end_point   = (int(end_point[0] + x1), int(end_point[1] + y1))
        return start_point, end_point

    def find_arrows(self, image_file):

        # Results
        arrows = []

        # Pre-Processing
        image = cv2.imread(image_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        ret,thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Arrow Detection
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(thresh.copy(), cv2.MORPH_OPEN, kernel, iterations=18)
        # cv2.imshow("process", closing)
        # cv2.waitKey(0)
        invert = cv2.bitwise_not(closing)
        img, contours, hierarchy = cv2.findContours(invert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.boxPoints(rect))
            start, end = self.__parse_arrow__(closing, box)
            arrows.append((start, end))
            # cv2.drawContours(image, [box], 0, (255, 0, 0), 2)
            cv2.circle(image, start, 10, (0, 0, 255), thickness=5)
            cv2.circle(image, end, 10, (255, 0, 0), thickness=5)
            cv2.putText(image, "Pointing From: " + str(start), (start[0], start[1] + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, "Pointing To: " + str(end), (end[0], end[1] + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Arrows", image)
        cv2.imwrite("arrows_main.png", image)
        cv2.waitKey(0)
        return arrows


detector = ArrowDetector()
print(detector.find_arrows("arrows4.png"))