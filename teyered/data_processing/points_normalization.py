import cv2
from imutils import face_utils
import numpy as np
import imutils
import dlib
import sys

# Universal size of the box for all normalized eye points
UNIVERSAL_RESIZE = 500  # [px]

def normalize_eye_points(facial_points_list):
    """
    Normalize eye points according to some universal measure for all photos.
    This allows the eye measurements to be compared.
    Main assumption is that the diameter of the eye (width) is const.
    This is based on a fact that only the eye height changes when the eye
    is being closed (no rotation of head is assumed)
    :param facial_points_list: List of tuples of points that describe a 
    particular facial feature 
    :return: Normalized facial points of the facial feature of the input 
    """
    if (facial_points_list == None or len(facial_points_list) <= 0):
        print("[teyered.data_processing.points_normalization." + 
              "normalize_eye_points] Facial points list is empty")

    # [(x1, y1), ..., (xn, yn)] - list of tuples of normalized facial points
    facial_points_list_normalized = []

    x_points = [x for (x, y) in facial_points_list]
    x_min = min(x_points)  # == 0
    x_max = max(x_points)  # == UNIVERSAL_RESIZE
    y_min = min([y for (x, y) in facial_points_list])  # == 0
    x_range = x_max - x_min  # This is always constant
    x_threshold = UNIVERSAL_RESIZE / x_range  # Value of x per one px

    # Parameter check
    if (x_min > x_max or x_range <= 0):
        print("[teyered.data_processing.points_normalization." + 
              "normalize_eye_points] Facial points list is invalid")

    # Normalize x and y
    for (x,y) in facial_points_list:
        facial_points_list_normalized.append(
            ((x - x_min) * x_threshold, ((y - y_min) / x_range) * 100)
            )
    
    return facial_points_list_normalized

# todo: rewrite the following tests in a separate directory

# Random sample points
test1 = [
    (10,10), 
    (15,7), 
    (18,5), 
    (21,9), 
    (17,12), 
    (13,18)
    ]
# Same area as test1 but the points are shifted by x + 5 and y + 2 
# (shifted left/right/up/down in the screen)
test2 = [
    (15,12),
    (20,9),
    (23,7),
    (26,11),
    (22,14),
    (18,20)
    ]
# Same area as test1, but the points are 10*size (closer to the screen)
test3 = [
    (100,100),
    (150,70), 
    (180,50), 
    (210,90), 
    (170,120), 
    (130,180)
    ]
# Area bigger than test1
test4 = [
    (10,10), 
    (15,3), 
    (18,2), 
    (21,11), 
    (17,25), 
    (13,30)
    ]

if __name__ == "__main__":
    print(normalize_eye_points(test1))
    print(normalize_eye_points(test2))
    print(normalize_eye_points(test3))
    print(normalize_eye_points(test4))