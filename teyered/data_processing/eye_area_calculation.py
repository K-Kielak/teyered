import logging

import cv2
import dlib
from imutils import face_utils, resize

from teyered.config import PREDICTOR_FILEPATH


logger = logging.getLogger(__name__)

# Teyered configuration
IMAGE_UPSAMPLE_FACTOR = 1  # Ease facial landmark detection (value from dlib)
UNIVERSAL_RESIZE = 500  # [px] Points are stored at this size
# Each detected point is given a label associated with a facial landmark
# Tuple marks lower and upper boundaries of the landmark labels
RIGHT_EYE_COORDINATES = (36, 42)
LEFT_EYE_COORDINATES = (42, 48)


def extract_facial_points(image):
    """
    Extract facial features and corresponding points from the input image
    :param image: Image frame numpy array to be processed
    :return: Facial features and corresponding points of the image
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_FILEPATH)
    facial_points = {'left_eye': [], 'right_eye': []}

    image = resize(image, width=UNIVERSAL_RESIZE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale image
    for (i, rect) in enumerate(detector(gray, IMAGE_UPSAMPLE_FACTOR)):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        for name, (i, j) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            for x, y in shape[i:j]:
                if (i, j) == RIGHT_EYE_COORDINATES:
                    facial_points['right_eye'].append((x, y))
                elif (i, j) == LEFT_EYE_COORDINATES:
                    facial_points['left_eye'].append((x, y))

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
    if not facial_points_list:
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
    if x_min > x_max or x_range <= 0:
        logger.warning('Facial points list is invalid')
        return None

    # Normalize x and y
    for x, y in facial_points_list:
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
        raise AttributeError('At least three points are needed '
                             'to calculate polygon area.')

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corner_points[i][0] * corner_points[j][1]
        area -= corner_points[j][0] * corner_points[i][1]

    return abs(area) / 2.0
