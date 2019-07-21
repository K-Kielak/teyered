import logging

import cv2
import numpy as np

from teyered.config import CAMERA_MATRIX, DIST_COEFFS, LEFT_EYE_COORDINATES, RIGHT_EYE_COORDINATES
from teyered.head_pose.pose import choose_pose_points


logger = logging.getLogger(__name__)


def project_eye_points(extracted_points_all, projection_points, r_vectors_all, t_vectors_all, camera_matrix = CAMERA_MATRIX, dist_coeffs = DIST_COEFFS):
    """
    Project eye points from world/model coordinates onto image coordinate system for all frames.
    :param frames: All frames to be analysed
    :param extracted_points_all: Scaled extracted eye points for all frames used to know whether or not to project on the frame
    :param projection_points: Eye points to be projected from world/model coordinate system to image coordinate system
    :param r_vectors_all: Rotation vectors for all frames
    :param t_vectors_all: Translation vectors for all frames
    :param camera_matrix: Default value from config.py
    :param dist_coeffs: Default value from config.py
    :return: np.ndarray of projected model points at each frame
    """
    if extracted_points_all.shape[0] != r_vectors_all.shape[0]:
        raise ValueError('r_vectors_all and extracted points array length must be the same')
    if extracted_points_all.shape[0] != t_vectors_all.shape[0]:
        raise ValueError('t_vectors_all and extracted points array length must be the same')
    if extracted_points_all.size < 1:
        raise ValueError('Must provide at least one frame for projection')

    projected_points_all = []
    
    for i in range(0, extracted_points_all.shape[0]):
        if extracted_points_all[i] is None:
            projected_points_all.append(None)
        else:
            projected_points, _ = cv2.projectPoints(projection_points, 
                r_vectors_all[i], t_vectors_all[i], camera_matrix, 
                dist_coeffs)
            projected_points_all.append(projected_points)
        
    return np.array(projected_points_all)

# Todo not sure why return type is tuple
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
    if frames.size < 1:
        raise ValueError('Must provide at least one frame')

    errors = []

    for i, frame in enumerate(frames):
        if extracted_points_all[i] is None:
            errors.append((i,0))
        else:
            # Sum of euclidian distances (todo, needs to be rewritten, there's np built-in function)
            error = np.sum(np.power(np.sum(np.power(extracted_points_all[i] - projected_points_all[i],2), axis=1), 0.5))
            errors.append((i, error))

    return np.array(errors)

def calculate_eye_closedness(extracted_points_all, projected_points_all):
    """
    Compare the projected eye points with extracted eye points and calculate closedness for all frames
    :param extracted_points_all: Extracted eye points for all frames
    :param projected_points_all: Projected eye points for all frames
    :return: np.ndarray of closedness percentage (1 == 100%) for all frames
    """
    if extracted_points_all.shape[0] != projected_points_all.shape[0]:
        raise ValueError('extracted_points_all and projected_points_all arrays must have the same length')
    if extracted_points_all.size < 1:
        raise ValueError('extracted_points_all and projected_points_all arrays must not be empty')

    eye_closedness = []

    for i in range(0, extracted_points_all.shape[0]):
        projected_points_sq = np.squeeze(projected_points_all[i])

        extracted_area = _calculate_polygon_area(extracted_points_all[i])
        projected_area = _calculate_polygon_area(projected_points_sq)
        
        # (x < 1) == more closed, (x > 1) == more open than usual
        eye_closedness.append(extracted_area / projected_area)

    return np.array(eye_closedness)

def choose_eye_points(facial_points):
    """
    Choose eye points from extracted facial points
    :param facial_points: Facial points extracted with points_extractor()
    :return: Eye points for left and right eyes
    """
    left_eye_points = np.array(facial_points[slice(*LEFT_EYE_COORDINATES)])
    right_eye_points = np.array(facial_points[slice(*RIGHT_EYE_COORDINATES)])
    return (left_eye_points, right_eye_points)


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
