import logging
import os
import sys

import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils


# Absolute path to the file directory
FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)
LOG_FILE_NAME = 'eye_area_calculation.log'
LOG_DIRECTORY = os.path.abspath(
    os.path.join(FILE_DIRECTORY, *[os.pardir, 'logs', LOG_FILE_NAME])
)
logger_formatter = logging.Formatter('%(asctime)s : %(name)s : %(message)s')
file_handler = logging.FileHandler(LOG_DIRECTORY)
file_handler.setFormatter(logger_formatter)
logger.addHandler(file_handler)

# Pictures are resized and points are stored at this width
RESIZE_WIDTH = 500  # [px]

PREDICTOR_FILENAME = 'shape_predictor_68_face_landmarks.dat'
PREDICTOR_FILEPATH = os.path.abspath(
    os.path.join(FILE_DIRECTORY,
        *[os.pardir, os.pardir, 'resources', PREDICTOR_FILENAME])
)
logger.debug(PREDICTOR_FILEPATH)

# Ease facial landmark detection (value from dlib usage example)
IMAGE_UPSAMPLE_FACTOR = 1  # clarification - upsample 1 time

# Each detected point is given a label associated with a facial landmark
# Tuple marks lower and upper boundaries of the landmark labels
RIGHT_EYE_COORDINATES = (36,42)
LEFT_EYE_COORDINATES = (42,48)

# Universal size of the box for all normalized eye points
UNIVERSAL_RESIZE = 500  # [px]


def extract_facial_points(image):
    """
    Extract facial features and corresponding points from the input image
    :param image: Image frame numpy array to be processed
    :return: Facial features and corresponding points of the image
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_FILEPATH)

    facial_points = { "left_eye" : [], "right_eye" : [] }

    image = imutils.resize(image, width=RESIZE_WIDTH)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    for (i, rect) in enumerate(detector(gray, IMAGE_UPSAMPLE_FACTOR)):

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            for (x, y) in shape[i:j]:
                if (i, j) == RIGHT_EYE_COORDINATES:
                    facial_points["right_eye"].append((x,y))
                elif (i, j) == LEFT_EYE_COORDINATES:
                    facial_points["left_eye"].append((x,y))

    logger.debug('Facial points were successfully extracted from the image')
    return facial_points


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
        logger.warning('Facial points list is empty')
        return None

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
        logger.warning('Facial points list is invalid')
        return None

    # Normalize x and y
    for (x,y) in facial_points_list:
        facial_points_list_normalized.append(
            ((x - x_min) * x_threshold, ((y - y_min) / x_range) * 100)
        )

    logger.debug('Facial points list normalized successfully')
    return facial_points_list_normalized


def calculate_polygon_area(corner_points):
    """
    Calculate area of the polygon from a given set of corner points
    :param corner_points: Corner points of the polygon
    :return: Area of the polygon
    """
    n = len(corner_points)
    if n < 3:
        logger.warning('At least three points are needed to calculate' +
            'polygon area')
        return None

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corner_points[i][0] * corner_points[j][1]
        area -= corner_points[j][0] * corner_points[i][1]
    area = abs(area) / 2.0
    return area
