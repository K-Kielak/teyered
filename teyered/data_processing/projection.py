import logging

import cv2
import numpy as np

from teyered.config import CAMERA_MATRIX, DIST_COEFFS


logger = logging.getLogger(__name__)


def project_points(projection_points, r_vectors_all, t_vectors_all,
                   camera_matrix=CAMERA_MATRIX, dist_coeffs=DIST_COEFFS):
    """
    Project model points from world/model coordinates onto image coordinate
    system for all frames
    :param projection_points: Points to be projected
    :param r_vectors_all: Rotation vectors for all frames
    :param t_vectors_all: Translation vectors for all frames
    :param camera_matrix: Intrinsic parameters
    :param dist_coeffs: Intrinsic parameters
    :return: np.ndarray of projected model points at each frame
    """
    if t_vectors_all.shape != r_vectors_all.shape:
        raise ValueError('t_vectors_all and r_vectors_all arrays length '
                         'must be the same')
    if t_vectors_all.size < 1:
        raise ValueError('Must provide at least one frame for projection')

    projected_points_all = []

    for i in range(0, t_vectors_all.shape[0]):
        if r_vectors_all[i] is None or t_vectors_all[i] is None:
            projected_points_all.append(None)
        else:
            projected_points, _ = cv2.projectPoints(projection_points,
                                                    r_vectors_all[i],
                                                    t_vectors_all[i],
                                                    camera_matrix,
                                                    dist_coeffs)
            
            projected_points = projected_points.reshape(
                (projected_points.shape[0], -1))
            projected_points_all.append(np.copy(projected_points))

    return np.array(projected_points_all)


def calculate_reprojection_error(ground_points_all, projected_points_all):
    """
    Calculate reprojection error based on euclidian distance between ground
    truth points and projected points
    :param ground_points_all: Ground truth points in all frames
    :param projected_points_all: Projected points in all frames
    :return: np.ndarray of float values of the reprojection error for each
    frame
    """
    if projected_points_all.shape != ground_points_all.shape:
        raise ValueError('projected_points_all and ground_points_all array '
                         'must have the same shape')
    if projected_points_all.size < 1:
        raise ValueError('Must provide at least one frame')

    errors = []

    for i in range(0, ground_points_all.shape[0]):
        if ground_points_all[i] is None or projected_points_all[i] is None:
            errors.append(-1)  # Nothing to project
        else:
            errors.append(np.linalg.norm(
                ground_points_all[i] - projected_points_all[i], axis=1).sum())

    return np.array(errors)
