import logging

from teyered.config import UNIVERSAL_RESIZE


logger = logging.getLogger(__name__)


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
        return []

    x_points = [x for (x, y) in facial_points_list]
    x_min = min(x_points)
    x_max = max(x_points)
    y_min = min([y for (x, y) in facial_points_list])
    x_range = x_max - x_min
    x_threshold = UNIVERSAL_RESIZE / x_range  # Value of x per one px

    assert x_min <= x_max, 'Facial points list is invalid'

    # Normalize x and y
    facial_points_list_normalized = []
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
