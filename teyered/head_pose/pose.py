import logging

import cv2
import numpy as np

from teyered.config import FACE_MODEL_FILEPATH, JAW_COORDINATES, \
     NOSE_COORDINATES, CAMERA_MATRIX, DIST_COEFFS


logger = logging.getLogger(__name__)


def _prepare_face_model():
    """
    Extract relevant features from the model file
    Model file: 68 points for x coord, 68 points for y coord, 68 points for
    z coord. Points match dlib facial features points
    :return: Array of tuples of face points in world coordinates
    """
    face_info = np.array(open(FACE_MODEL_FILEPATH, "r+").readlines(),
                         dtype=np.float32)

    face_model = np.zeros((68, 3), dtype=np.float32)
    face_model[:,0] = face_info[0:68]  # x coordinate
    face_model[:,1] = (-1) * face_info[68:136]  # y coordinate (reverse)
    face_model[:,2] = (-1) * face_info[136:204]  # z coordinate (reverse)

    return _choose_pose_points(face_model)

def _choose_pose_points(facial_points):
    """
    Choose facial points which will be used to solve PnP
    :param facial_points: array of facial points from detection algorithm
    :return: array of tuples for specific facial points
    """
    points_jaw = facial_points[slice(*JAW_COORDINATES)]
    points_nose = facial_points[slice(*NOSE_COORDINATES)]
    points_r_eye = [facial_points[i] for i in [36, 39]]  # Corners
    points_l_eye = [facial_points[i] for i in [42, 45]]  # Corners
    points_mouth = [facial_points[i] for i in [48, 54]]  # Corners

    points = np.concatenate((points_jaw, points_nose, points_r_eye,
                             points_l_eye, points_mouth))

    return np.array([tuple(p) for p in points], dtype='double')

def _get_euler_angles(rotation_matrix, translation_vector):
    """
    Get euler angles from rotation matrix and translation vector (XYZ)
    """
    extrinsic_parameters = np.hstack((rotation_matrix, translation_vector))
    projection_matrix = CAMERA_MATRIX.dot(extrinsic_parameters)
    euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)[-1]

    yaw = euler_angles[1]
    pitch = euler_angles[0]
    roll = euler_angles[2]

    return (yaw, pitch, roll)

def _get_rotation_matrix(rotation_vector):
    """
    Convert axis and angle of rotation representation to rotation matrix
    """
    return cv2.Rodrigues(rotation_vector)

def _solve_pnp(image_points, model_points, prev_rvec = None,
                     prev_tvec = None):
    """
    Calculate rotation and translation vectors by solving PnP
    :param image_points: Numpy array of image points at this specific frame
    :param model_points: Numpy array of face model in world coordinates
    :param prev_rvec: Previous estimated rotation vector
    :param prev_tvec: Previous estimated translation vector
    :return: Tuple of rotation and translation vectors for the frame
    """
    if prev_rvec is None or prev_tvec is None:
        _, r_vector, t_vector = cv2.solvePnP(model_points, image_points,
                                             CAMERA_MATRIX, DIST_COEFFS,
                                             flags=cv2.cv2.SOLVEPNP_ITERATIVE)
    else:
        _, r_vector, t_vector = cv2.solvePnP(model_points, image_points,
                                             CAMERA_MATRIX, DIST_COEFFS,
                                             rvec=prev_rvec, tvec=prev_tvec,
                                             useExtrinsicGuess=True,
                                             flags=cv2.cv2.SOLVEPNP_ITERATIVE)
    return (r_vector, t_vector)

def _get_camera_world_coord(rotation_matrix, t_vector):
    """
    Use object's rotation matrix and translation vector to calculate camera's
    position in world coordinates
    """
    camera_pose_world = -np.matrix(rotation_matrix).T * np.matrix(t_vector)
    return camera_pose_world.reshape(1, -1)

def estimate_pose(facial_points_all):
    """
    Estimate 3D pose of an object in camera coordinates from given facial points
    :param facial_points_all: List of facial points coordinates for all frames
    :return: 4-tuple containing rotation vector, translation vector,
    euler angles and camera position in world coordinates for every frame
    """
    # Information to be returned
    r_vectors_all = []
    t_vectors_all = []
    angles_all = []
    camera_world_coord_all = []

    # Variables for PnP calculation based on previous calculations
    prev_rvec = None
    prev_tvec = None

    model_points = _prepare_face_model()
    logger.debug('Face model points have been prepared successfully')

    for facial_points in facial_points_all:
        # No facial points were detected for that frame, skip and reset
        if facial_points.shape == (0,):
            r_vectors_all.append([])
            t_vectors_all.append([])
            angles_all.append([])
            camera_world_coord_all.append([])
            prev_rvec = None
            prev_tvec = None
            continue

        facial_points = _choose_pose_points(facial_points)

        r_vector, t_vector = _solve_pnp(facial_points, model_points,
                                        prev_rvec, prev_tvec)
        prev_rvec = r_vector
        prev_tvec = t_vector

        rotation_matrix, _ = _get_rotation_matrix(r_vector)
        yaw, pitch, roll = _get_euler_angles(rotation_matrix, t_vector)
        camera_world_coord = _get_camera_world_coord(rotation_matrix, t_vector)

        # The following will use np.copy() since otherwise list is always
        # appended by the same reference for some reason, and all array
        # elements ends up pointing to the same reference
        r_vectors_all.append(np.copy(r_vector))
        t_vectors_all.append(np.copy(t_vector))
        angles_all.append((yaw, pitch, roll))
        camera_world_coord_all.append(camera_world_coord)

    logger.debug('Shape estimation has finished successfully')
    return (r_vectors_all, t_vectors_all, angles_all, camera_world_coord_all)

def estimate_pose_live(facial_points, prev_rvec, prev_tvec, model_points):
    facial_points = _choose_pose_points(facial_points)

    r_vector, t_vector = _solve_pnp(facial_points, model_points,
                                    prev_rvec, prev_tvec)

    rotation_matrix, _ = _get_rotation_matrix(r_vector)
    yaw, pitch, roll = _get_euler_angles(rotation_matrix, t_vector)
    camera_world_coord = _get_camera_world_coord(rotation_matrix, t_vector)

    return (r_vector, t_vector, [yaw, pitch, roll], camera_world_coord)
