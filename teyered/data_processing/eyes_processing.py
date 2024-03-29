import logging

import numpy as np

from teyered.config import CAMERA_MATRIX, DIST_COEFFS, LEFT_EYE_COORDINATES
from teyered.config import RIGHT_EYE_COORDINATES
from teyered.data_processing.projection import project_points


logger = logging.getLogger(__name__)


def calculate_eye_closedness(extracted_points_all, model_points,
                             r_vectors_all, t_vectors_all,
                             camera_matrix=CAMERA_MATRIX,
                             dist_coeffs=DIST_COEFFS):
    """
    Calculate eye closedness for all frames
    TODO Detect closedness of a single eye if data is not available 
    for the other
    :param extracted_points_all: Extracted facial points for all frames
    :param model_points: Model points in 3D world coordinates
    :param r_vectors_all: Rotation vectors for all frames
    :param t_vectors_all: Translation vectors for all frames
    :param camera_matrix: Intrinsic parameters
    :param dist_coeffs: Intrinsic parameters
    :return: np.ndarray of either eye closedness ratio or value -1 if
    it cannot be determined 
    """
    if extracted_points_all.shape[0] != r_vectors_all.shape[0]:
        raise ValueError('extracted_points_all and r_vectors_all arrays '
                         'length must be the same')
    if extracted_points_all.shape[0] != t_vectors_all.shape[0]:
        raise ValueError('extracted_points_all and t_vectors_all arrays '
                         'length must be the same')
    if extracted_points_all.size < 1:
        raise ValueError('Must provide at least one frame')

    # Get eye points from extrated points and model points
    extracted_eye_points_left_all, extracted_eye_points_right_all = \
        _choose_eye_points(extracted_points_all)
    projected_eye_points_left_all, projected_eye_points_right_all = \
        _project_eye_points(model_points, r_vectors_all, t_vectors_all)

    # Calculate closedness for each frame
    eye_closedness_left_all = []
    eye_closedness_right_all = []

    for i in range(0, extracted_eye_points_left_all.shape[0]):
        if extracted_eye_points_left_all[i] is None or \
           projected_eye_points_left_all[i] is None or \
           extracted_eye_points_right_all[i] is None or \
           projected_eye_points_right_all[i] is None:
            eye_closedness_left_all.append(-1)  # Nothing to project
            eye_closedness_right_all.append(-1)  # Nothing to project
            continue

        extracted_area_left = _calculate_polygon_area(
            extracted_eye_points_left_all[i]
        )
        projected_area_left = _calculate_polygon_area(
            projected_eye_points_left_all[i]
        )

        extracted_area_right = _calculate_polygon_area(
            extracted_eye_points_right_all[i]
        )
        projected_area_right = _calculate_polygon_area(
            projected_eye_points_right_all[i]
        )

        if projected_area_left == 0 or projected_area_right == 0:
            eye_closedness_left_all.append(-1)
            eye_closedness_right_all.append(-1)
            continue

        # (x < 1) == more closed, (x > 1) == more open than usual, 1 == 100%
        eye_closedness_left_all.append(
            extracted_area_left / projected_area_left
        )
        eye_closedness_right_all.append(
            extracted_area_right / projected_area_right
        )

    logger.debug('Eye closedess calculated successfully')
    return (np.array(eye_closedness_left_all),
            np.array(eye_closedness_right_all))


def _project_eye_points(model_points, r_vectors_all, t_vectors_all,
                        camera_matrix=CAMERA_MATRIX, 
                        dist_coeffs=DIST_COEFFS):
    """
    Project eye points from model points
    :param model_points: Model points in 3D world coordinates
    :param r_vectors_all: Rotation vectors for all frames
    :param t_vectors_all: Translation vectors for all frames
    :param camera_matrix: Intrinsic parameters
    :param dist_coeffs: Intrinsic parameters
    :return: np.ndarray of projected points for both eyes for all frames
    """
    eye_model_points_left, eye_model_points_right = \
        _choose_eye_points(np.array([model_points]))
    eye_model_points_left = eye_model_points_left[0]
    eye_model_points_right = eye_model_points_right[0]

    # Project eye model points onto image plane
    projected_eye_points_left_all = project_points(eye_model_points_left,
                                                   r_vectors_all,
                                                   t_vectors_all,
                                                   camera_matrix=camera_matrix,
                                                   dist_coeffs=dist_coeffs)
    projected_eye_points_right_all = project_points(eye_model_points_right,
                                                    r_vectors_all,
                                                    t_vectors_all,
                                                    camera_matrix=camera_matrix,
                                                    dist_coeffs=dist_coeffs)
    
    return projected_eye_points_left_all, projected_eye_points_right_all


def _choose_eye_points(facial_points_all):
    """
    Choose left and right eye points from facial points
    :param facial_points_all: Facial points adhering to face model
    requirements for all frames
    :return: np.ndarray of eye points for left and right eyes for all frames
    """
    if facial_points_all.shape[0] < 1:
        raise ValueError('Must provide at least one frame')

    left_eye_points_all = []
    right_eye_points_all = []

    for facial_points in facial_points_all:
        if facial_points is None:
            left_eye_points_all.append(None)
            right_eye_points_all.append(None)
        else:
            left_eye_points_all.append(
                np.array(facial_points[slice(*LEFT_EYE_COORDINATES)])
            )
            right_eye_points_all.append(
                np.array(facial_points[slice(*RIGHT_EYE_COORDINATES)])
            )

    return np.array(left_eye_points_all), np.array(right_eye_points_all)


def _calculate_polygon_area(corner_points):
    """
    Calculate area of the polygon from a given set of corner points
    :param corner_points: np.ndarray of corner points of the polygon
    :return: Area of the polygon
    """
    n = corner_points.shape[0]
    if n < 3:
        raise ValueError('At least three points are needed '
                         'to calculate polygon area.')

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corner_points[i][0] * corner_points[j][1]
        area -= corner_points[j][0] * corner_points[i][1]

    return abs(area) / 2.0
