import logging

import cv2
import numpy as np

from teyered.config import IMAGE_UPSAMPLE_FACTOR, UNIVERSAL_RESIZE, \
     FACE_MODEL_FILEPATH, JAW_COORDINATES, NOSE_COORDINATES


logger = logging.getLogger(__name__)

# Constants
TRACKING_LENGTH = 5  # frame

# Camera parameters (ideally from a calibrated camera)
focal_length = UNIVERSAL_RESIZE
center = (UNIVERSAL_RESIZE/2, 333/2)
camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                         )
dist_coeffs = np.zeros((4,1))


def _prepare_face_model():
    """
    Extract relevant features from the model file
    Model file: 68 points for x coord, 68 points for y coord, 68 points for
    z coord. Points match dlib facial features points
    :return: array of tuples of face points in world coordinates
    """
    face_info = np.array(open(FACE_MODEL_FILEPATH, "r+").readlines(),
                         dtype=np.float32)

    face_model = np.zeros((68,3), dtype=np.float32)
    face_model[:,0] = face_info[0:68]  # x coordinate
    face_model[:,1] = (-1)*face_info[68:136]  # y coordinate (reverse)
    face_model[:,2] = (-1)*face_info[136:204]  # z coordinate (reverse)

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
    projection_matrix = camera_matrix.dot(extrinsic_parameters)
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
    :param image_points: numpy array of image points at this specific frame
    :param model_points: numpy array of face model in world coordinates
    :param prev_rvec: previous estimated rotation vector
    :param prev_tvec: previous estimated translation vector
    :return: tuple of rotation and translation vector for the frame
    """
    if prev_rvec == None or prev_tvec == None:
        _, r_vector, t_vector = cv2.solvePnP(model_points, image_points,
                                             camera_matrix, dist_coeffs,
                                             flags=cv2.cv2.SOLVEPNP_ITERATIVE)
    else:
        _, r_vector, t_vector = cv2.solvePnP(model_points, image_points,
                                             camera_matrix, dist_coeffs,
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
    return camera_pose_world.reshape(1,-1)

def estimate_pose(facial_points_all):
    """
    Estimate 3D pose of an object in camera coordinates from given facial points
    over time
    :param facial_points_all: facial points coordinates for all frames
    :return: 4-tuple containing rotation vector, translation vector,
    euler angles and camera position in world coordinates
    coordinates for every frame
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
        if not facial_points:
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

        rotation_matrix, _ = _get_rotation_matrix(r_vector)
        yaw, pitch, roll = _get_euler_angles(rotation_matrix, t_vector)
        camera_world_coord = _get_camera_world_coord(rotation_matrix, t_vector)

        r_vectors_all.append(r_vector)
        t_vectors_all.append(t_vector)
        angles_all.append((yaw, pitch, roll))
        camera_world_coord_all.append(camera_world_coord)

        prev_rvec = r_vector
        prev_tvec = t_vector

    logger.debug('Shape estimation has finished successfully')
    return (np.array(r_vectors_all), np.array(t_vectors_all),
            np.array(angles_all), np.array(camera_world_coord_all))
