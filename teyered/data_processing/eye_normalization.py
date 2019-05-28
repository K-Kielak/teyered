import logging

import cv2
import numpy as np

from teyered.config import CAMERA_MATRIX, DIST_COEFFS
from teyered.head_pose.pose import choose_pose_points

logger = logging.getLogger(__name__)


def project_eye_points_live(model_points, rotation_vector, translation_vector):
    """
    Project points from world/model coordinates onto image coordinate system in a single frame
    :param model_points: Model points to be projected
    :param rotation_vector:
    :param translation_vector:
    :return: Projected points in image plane coordinates
    """
    model_points_projected, _ = cv2.projectPoints(model_points, 
        rotation_vector, translation_vector, CAMERA_MATRIX, 
        DIST_COEFFS)

    return model_points_projected

def project_eye_points(frames, facial_points_all, model_points, r_vectors_all, t_vectors_all):
    """
    Project points from world/model coordinates onto image coordinate system for all frames
    :param frames: All frames
    :param facial_points_all: Facial points in all frames (currently to see which ones weren't detected)
    :param model_points: Model points to be projected
    :param r_vectors_all: Rotation vectors for each frame
    :param t_vectors_all: Translation vectors for each frame
    :return: Array of projected model points at each frame
    """
    model_points_projected_all = []
    
    frame_count = 0
    for frame in frames:
        if facial_points_all[frame_count] == []:
            model_points_projected_all.append([])
            frame_count += 1
            continue

        model_points_projected, _ = cv2.projectPoints(model_points, 
            r_vectors_all[frame_count], t_vectors_all[frame_count], CAMERA_MATRIX, 
            DIST_COEFFS)

        model_points_projected_all.append(model_points_projected)
        frame_count += 1

    return np.array(model_points_projected_all)

def calculater_reprojection_error_live(facial_points, reprojected_points):
    """
    Calculate reprojection error based on euclidian distance between points
    :param facial_points: Facial points used in pose estimation
    :param reprojected_points: Reprojected 3D model points used in pose estimation
    :return: Float value of the overall error
    """
    return np.sum(np.power(np.sum(np.power(facial_points-reprojected_points,2), axis=1), 0.5))

def calculater_reprojection_error(frames, facial_points_all, reprojected_points_all):
    """
    Calculate reprojection error based on euclidian distance between points. Difference from live version is that the pose points are chosen here instead of being already passed as a parameter
    :param frames: All frames
    :param facial_points_all: All facial points in all frames
    :param reprojected_points_all: All reprojected points from 3D model in all frames
    :return: A list of float values of the overall error (ready to be saved in a .csv file using file.py save_points method)
    """
    errors = []

    frame_count = 0
    for frame in frames:
        if facial_points_all[frame_count] == []:
            errors.append((frame_count,0))
            frame_count += 1
            continue

        error = np.sum(np.power(np.sum(np.power(choose_pose_points(facial_points_all[frame_count]) - choose_pose_points(reprojected_points_all[frame_count]),2), axis=1), 0.5))

        errors.append((frame_count, error))
        frame_count += 1

    return errors

def compare_projected_facial(model_points_projected, facial_points):
    """
    Compare the projected points with obtained facial points
    """
    
    model_points_projected = np.squeeze(model_points_projected)

    projected_area = _calculate_polygon_area(model_points_projected)
    obtained_area = _calculate_polygon_area(facial_points)

    # (closedness < 1) = more closed, (closedness > 1) = more open than usual
    closedness = obtained_area / projected_area

    return closedness

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
