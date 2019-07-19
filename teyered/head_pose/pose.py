import logging

import cv2
import numpy as np

from teyered.config import FACE_MODEL_FILEPATH, JAW_COORDINATES, \
     NOSE_COORDINATES, CAMERA_MATRIX, DIST_COEFFS


logger = logging.getLogger(__name__)


def choose_pose_points(facial_points):
    """
    Choose facial points which will be used to solve PnP
    :param facial_points: Array of facial points from detection algorithm
    :return: np.ndarray of tuples for specific facial points
    """
    points_jaw = facial_points[slice(*JAW_COORDINATES)]
    points_nose = facial_points[slice(*NOSE_COORDINATES)]
    points_r_eye = [facial_points[i] for i in [36, 39]]  # Corners
    points_l_eye = [facial_points[i] for i in [42, 45]]  # Corners
    points_mouth = [facial_points[i] for i in [48, 54]]  # Corners

    points = np.concatenate((points_jaw, points_nose, points_r_eye,
                             points_l_eye, points_mouth))

    return np.array([tuple(p) for p in points], dtype='double')

def get_euler_angles(rotation_matrix, translation_vector):
    """
    Get euler angles from rotation matrix and translation vector (XYZ)
    :param rotation_matrix: Rotation matrix from get_rotation_matrix()
    :param translation_vector: Translation vector from solve_pnp()
    :return: Yaw, pitch and roll angles in this specific order
    """
    extrinsic_parameters = np.hstack((rotation_matrix, translation_vector))
    projection_matrix = CAMERA_MATRIX.dot(extrinsic_parameters)
    euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)[-1]

    yaw = euler_angles[1]
    pitch = euler_angles[0]
    roll = euler_angles[2]

    return (yaw, pitch, roll)

def get_rotation_matrix(rotation_vector):
    """
    Convert axis and angle of rotation representation to rotation matrix
    :param rotation_vector: Rotation vector from solve_pnp()
    :return: np.ndarray rotation matrix
    """
    return cv2.Rodrigues(rotation_vector)

def solve_pnp(image_points, model_points, prev_rvec = None,
                     prev_tvec = None):
    """
    Calculate rotation and translation vectors by solving PnP
    :param image_points: Numpy array of image points at this specific frame
    :param model_points: Numpy array of face model in world coordinates
    :param prev_rvec: Previous estimated rotation vector
    :param prev_tvec: Previous estimated translation vector
    :return: np.ndarray rotation and translation vectors for the frame
    """
    if prev_rvec is None or prev_tvec is None:
        _, r_vector, t_vector, inliers = cv2.solvePnPRansac(model_points, image_points,
                                             CAMERA_MATRIX, DIST_COEFFS, useExtrinsicGuess=False,
                                             flags=cv2.cv2.SOLVEPNP_ITERATIVE)
        """
        _, r_vector, t_vector = cv2.solvePnP(model_points, image_points,
                                             CAMERA_MATRIX, DIST_COEFFS,
                                             flags=cv2.cv2.SOLVEPNP_ITERATIVE)
        """
    else:
        _, r_vector, t_vector, inliers = cv2.solvePnPRansac(model_points, image_points,
                                        CAMERA_MATRIX, DIST_COEFFS, rvec=prev_rvec, tvec=prev_tvec, useExtrinsicGuess=True,
                                        flags=cv2.cv2.SOLVEPNP_ITERATIVE)
        """
        _, r_vector, t_vector = cv2.solvePnP(model_points, image_points,
                                             CAMERA_MATRIX, DIST_COEFFS,
                                             rvec=prev_rvec, tvec=prev_tvec,
                                             useExtrinsicGuess=True,
                                             flags=cv2.cv2.SOLVEPNP_ITERATIVE)
        """
    return (r_vector, t_vector)

def get_camera_world_coord(rotation_matrix, t_vector):
    """
    Use object's rotation matrix and translation vector to calculate camera's position in world coordinates
    :param rotation_matrix: Rotation matrix from get_rotation_matrix()
    :param t_vector: Translation vector from solve_pnp()
    :return: np.ndarray of camera's position in world coordinates
    """
    camera_pose_world = -np.matrix(rotation_matrix).T * np.matrix(t_vector)
    return camera_pose_world.reshape(1, -1)

def estimate_pose(facial_points_all, model_points, prev_rvec = None, prev_tvec = None):
    """
    Estimate 3D pose of an object in camera coordinates from given facial points
    :param facial_points_all: Scaled facial points coordinates for all frames
    :param model_points: Model points of the face in world/model coordinates
    :param prev_rvec: Previous rotation vector if this is not the beginning of the video
    :param prev_tvec: Previous translation vector if this is not the beginning of the video
    :return: Rotation and translation vectors, euler angles and camera position in world coordinates for every frame as np.ndarray
    """
    if (prev_rvec is not None and prev_tvec is None) or (prev_rvec is None and prev_tvec is not None):
        raise ValueError('Previous rotation and translation vectors must be provided together')

    # Information to be returned
    r_vectors_all = []
    t_vectors_all = []
    angles_all = []
    camera_world_coord_all = []

    # Choose model points
    model_points_pose = choose_pose_points(model_points)

    for facial_points in facial_points_all:
        # No facial points were detected for that frame, skip and reset
        if not facial_points:
            r_vectors_all.append(None)
            t_vectors_all.append(None)
            angles_all.append(None)
            camera_world_coord_all.append(None)
            prev_rvec = None
            prev_tvec = None
            continue

        # This may be a bottleneck here, maybe apply on the whole facial_points_all as filter function?
        facial_points_pose = choose_pose_points(facial_points)

        r_vector, t_vector = solve_pnp(facial_points_pose, model_points_pose,
                                        prev_rvec, prev_tvec)
        prev_rvec = r_vector
        prev_tvec = t_vector

        rotation_matrix, _ = get_rotation_matrix(r_vector)
        yaw, pitch, roll = get_euler_angles(rotation_matrix, t_vector)
        camera_world_coord = get_camera_world_coord(rotation_matrix, t_vector)

        # The following will use np.copy() since otherwise list is always
        # appended by the same reference for some reason, and all array
        # elements ends up pointing to the same reference
        r_vectors_all.append(np.copy(r_vector))
        t_vectors_all.append(np.copy(t_vector))
        angles_all.append(np.array([yaw, pitch, roll]))
        camera_world_coord_all.append(camera_world_coord)

    logger.debug('Shape estimation has finished successfully')
    return (np.array(r_vectors_all), np.array(t_vectors_all), np.array(angles_all), np.array(camera_world_coord_all))

def estimate_pose_live(facial_points, model_points, prev_rvec = None, prev_tvec = None):
    """
    Live version of estimate_pose()
    :param facial_points: Facial points for the frame
    :param model_points: Model points for the frame
    :param prev_rvec: Previous rotation vector
    :param prev_tvec: Previous translation vector
    :return: Rotation and translation vectors, angles and camera in world coordinates as np.ndarray
    """
    if (prev_rvec is not None and prev_tvec is None) or (prev_rvec is None and prev_tvec is not None):
        raise ValueError('Previous rotation and translation vectors must be provided together')

    facial_points_pose = choose_pose_points(facial_points)
    model_points_pose = choose_pose_points(model_points)

    r_vector, t_vector = solve_pnp(facial_points_pose, model_points_pose,
                                    prev_rvec, prev_tvec)

    rotation_matrix, _ = get_rotation_matrix(r_vector)
    yaw, pitch, roll = get_euler_angles(rotation_matrix, t_vector)
    camera_world_coord = get_camera_world_coord(rotation_matrix, t_vector)

    return (r_vector, t_vector, np.array([yaw, pitch, roll]), camera_world_coord)
