import logging

import cv2

from teyered.config import UNIVERSAL_RESIZE


logger = logging.getLogger(__name__)


def draw_pose(frames, facial_points_all, r_vectors_all, t_vectors_all,
              camera_matrix, dist_coeffs):
    """
    Draw 3D pose estimation using basis vectors projected onto image coordinate
    system
    :param frames: all frames of the video
    :param facial_points_all: facial points in each frame (all of them)
    :param r_vectors_all: rotation vectors in each frame
    :param t_vectors_all: translation vectors in each frame
    :param camera_matrix: camera matrix
    :param dist_coeffs: distortion coefficients matrix
    :return: frames with pose drawn on them
    """
    if len(frames) != len(facial_points_all) or \
       len(frames) != len(r_vectors_all) or len(frames) != len(t_vectors_all):
       logger.warning('Facial points, rotation vectors and translation' + \
        'vectors must be provided for every frame')

    for i in range(0, len(frames)):
        rot_x, _ = cv2.projectPoints(np.array([(100.0, 0.0, 0.0)]),
                                     r_vectors_all[i], t_vectors_all[i],
                                     camera_matrix, dist_coeffs)
        rot_y, _ = cv2.projectPoints(np.array([(0.0, 100.0, 0.0)]),
                                     r_vectors_all[i], t_vectors_all[i],
                                     camera_matrix, dist_coeffs)
        rot_z, _ = cv2.projectPoints(np.array([(0.0, 0.0, 100.0)]),
                                     r_vectors_all[i], t_vectors_all[i],
                                     camera_matrix, dist_coeffs)

        p_origin = (int(facial_points_all[i][30][0]),
                    int(facial_points_all[i][30][1]))
        p_x = ( int(rot_x[0][0][0]), int(rot_x[0][0][1]))
        p_y = ( int(rot_y[0][0][0]), int(rot_y[0][0][1]))
        p_z = ( int(rot_z[0][0][0]), int(rot_z[0][0][1]))

        cv2.line(frames[i], p_origin, p_x, (255,0,0), 2) # B
        cv2.line(frames[i], p_origin, p_y, (0,255,0), 2) # G
        cv2.line(frames[i], p_origin, p_z, (0,0,255), 2) # R

    logger.debug('Pose vectors drawn successfully')
    return frames

def draw_facial_points(frames, facial_points_all):
    """
    :param frames: frames scaled to facial points
    :param facial_points: array of facial points to be drawn for each frame
    :return: frames with marked facial points
    """
    if len(frame) != len(facial_points_all):
        logger.warning('Facial points must be provided for every frame')

    for i in range(0, len(frames)):
        for point in facial_points[i]:
            cv2.circle(frames[i], (point[0], point[1]), 1, (0, 255, 0), -1)

    logger.debug('Facial points drawn successfully')
    return frames

def gray_and_resize(image):
    """
    Turn image to grayscale and resize to UNIVERSAL_RESIZE
    """
    image = resize(image, width=UNIVERSAL_RESIZE)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
