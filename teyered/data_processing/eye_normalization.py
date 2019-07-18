import logging

import cv2
import numpy as np

from teyered.config import CAMERA_MATRIX, DIST_COEFFS
from teyered.head_pose.pose import choose_pose_points


logger = logging.getLogger(__name__)


def project_eye_points(frames, extracted_points_all, projection_points, r_vectors_all, t_vectors_all, camera_matrix = CAMERA_MATRIX, dist_coeffs = DIST_COEFFS):
    """
    Project eye points from world/model coordinates onto image coordinate system for all frames.
    :param frames: All frames to be analysed
    :param extracted_points_all: Scaled extracted eye points for all frames
    :param projection_points: Eye points to be projected from world/model coordinate system to image coordinate system
    :param r_vectors_all: Rotation vectors for all frames
    :param t_vectors_all: Translation vectors for all frames
    :param camera_matrix: Default value from config.py
    :param dist_coeffs: Default value from config.py
    :return: np.ndarray of projected model points at each frame
    """
    if frames.shape[0] != extracted_points_all.shape[0]:
        raise ValueError('extracted_points_all and frames array length must be the same')
    if frames.shape[0] != r_vectors_all.shape[0]:
        raise ValueError('r_vectors_all and frames array length must be the same')
    if frames.shape[0] != t_vectors_all.shape[0]:
        raise ValueError('t_vectors_all and frames array length must be the same')
    if not frames.size:
        raise ValueError('frames array size cannot be 0')

    projected_points_all = []
    
    for i, frame in enumerate(frames):
        if extracted_points_all[i] is None:
            projected_points_all.append(None)
        else:
            projected_points, _ = cv2.projectPoints(projection_points, 
                r_vectors_all[i], t_vectors_all[i], camera_matrix, 
                dist_coeffs)
            projected_points_all.append(projected_points)
        
    return np.array(projected_points_all)

def project_eye_points_live(projection_points, r_vector, t_vector, camera_matrix = CAMERA_MATRIX, dist_coeffs = DIST_COEFFS):
    """
    Live version of project_eye_points().
    :param projection_points: Eye points to be projected from world/model coordinate system to image coordinate system
    :param r_vector: Rotation vector
    :param t_vector: Translation vector
    :param camera_matrix: Default value from config.py
    :param dist_coeffs: Default value from config.py
    :return: np.ndarray of projected points in image plane coordinates
    """
    projected_points, _ = cv2.projectPoints(projection_points, 
        r_vector, t_vector, camera_matrix, dist_coeffs)

    return projected_points

def calculate_reprojection_error(frames, extracted_points_all, projected_points_all):
    """
    Calculate reprojection error based on euclidian distance between extracted points and projected points. Extracted and projected points must be the same points
    :param frames: All frames to be analysed
    :param extracted_points_all: Extracted points in all frames
    :param reprojected_points_all: Projected points in all frames
    :return: np.ndarray of float values of the overall reprojection error (to be saved in csv using save_points() from file.py)
    """
    if frames.shape[0] != extracted_points_all.shape[0]:
        raise ValueError('extracted_points_all and frames array length must be the same')
    if frames.shape[0] != projected_points_all.shape[0]:
        raise ValueError('projected_points_all and frames array length must be the same')
    if not frames.size:
        raise ValueError('frames array size cannot be 0')

    errors = []

    for i, frame in enumerate(frames):
        if not extracted_points_all[i]:
            errors.append((i,0))
        else:
            # Sum of euclidian distances (todo, needs to be rewritten, there's np built-in function)
            error = np.sum(np.power(np.sum(np.power(extracted_points_all[i] - projected_points_all[i],2), axis=1), 0.5))
            errors.append((i, error))

    return np.array(errors)

def calculate_reprojection_error_live(extracted_points, projected_points):
    """
    Live version of calculate_reprojection_error(). Extracted and projected points must be the same points
    :param extracted_points: Extracted points
    :param projected_points: Projected points
    :return: Float value of the overall reprojection error
    """
    if extracted_points.shape != reprojected_points.shape:
        raise ValueError('extracted_points and reprojected_points arrays must have the same shape')

    # Sum of euclidian distances (todo, needs to be rewritten, there's np built-in function)
    return np.sum(np.power(np.sum(np.power(extracted_points-reprojected_points,2), axis=1), 0.5))

def calculate_eye_closedness(extracted_points_all, projected_points_all):
    """
    Compare the projected eye points with extracted eye points and calculate closedness for all frames
    :param extracted_points_all: Extracted eye points for all frames
    :param projected_points_all: Projected eye points for all frames
    :return: np.ndarray of closedness percentage (1 == 100%) for all frames
    """
    if extracted_points_all.shape[0] != projected_points_all.shape[0]:
        raise ValueError('extracted_points_all and projected_points_all arrays must have the same length')
    if not extracted_points_all.size:
        raise ValueError('extracted_points_all and projected_points_all arrays must not be empty')

    eye_closedness = []

    for i in range(0, extracted_points_all.shape[0]):
        projected_points_sq = np.squeeze(projected_points_all[i])

        extracted_area = _calculate_polygon_area(extracted_points_all[i])
        projected_area = _calculate_polygon_area(projected_points_sq)
        
        # (x < 1) == more closed, (x > 1) == more open than usual
        eye_closedness.append(extracted_area / projected_area)

    return np.array(eye_closedness)

def calculate_eye_closedness_live(extracted_points, projected_points):
    """
    Live version of calculate_eye_closedness()
    :param extracted_points: Extracted eye points
    :param projected_points: Projected eye points
    :return: Closedness percentage (1 == 100%)
    """
    projected_points_sq = np.squeeze(projected_points)

    extracted_area = _calculate_polygon_area(extracted_points)
    projected_area = _calculate_polygon_area(projected_points_sq)

    # (x < 1) == more closed, (x > 1) == more open than usual
    return extracted_area / projected_area

def _calculate_polygon_area(corner_points):
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
