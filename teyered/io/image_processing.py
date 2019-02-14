import logging

import cv2
from imutils import resize
import numpy as np

from teyered.config import UNIVERSAL_RESIZE


logger = logging.getLogger(__name__)


def draw_pose(frames, facial_points_all, r_vectors_all, t_vectors_all,
              camera_matrix, dist_coeffs):
    """
    Draw 3D pose estimation using basis vectors projected onto image coordinate
    system
    :param frames: All frames of the video scaled to other parameters
    :param facial_points_all: Facial points in each frame
    :param r_vectors_all: Rotation vectors in each frame
    :param t_vectors_all: Translation vectors in each frame
    :param camera_matrix: Camera matrix
    :param dist_coeffs: Distortion coefficients matrix
    :return: All frames with pose drawn on them
    """
    edited_frames = []

    if len(frames) != len(facial_points_all) or \
       len(frames) != len(r_vectors_all) or len(frames) != len(t_vectors_all):
       logger.warning('Facial points, rotation vectors and translation' \
             'vectors must be provided for every frame')
       return None

    for i in range(0, len(frames)):
        edited_frame = frames[i]

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
        p_x = (int(rot_x[0][0][0]), int(rot_x[0][0][1]))
        p_y = (int(rot_y[0][0][0]), int(rot_y[0][0][1]))
        p_z = (int(rot_z[0][0][0]), int(rot_z[0][0][1]))

        cv2.line(edited_frame, p_origin, p_x, (255, 0, 0), 2) # B
        cv2.line(edited_frame, p_origin, p_y, (0, 255, 0), 2) # G
        cv2.line(edited_frame, p_origin, p_z, (0, 0, 255), 2) # R

        edited_frames.append(edited_frame)

    logger.debug('Pose vectors drawn successfully')
    return np.array(edited_frames)

def draw_facial_points(frames, facial_points_all):
    """
    :param frames: All frames scaled to facial points
    :param facial_points: Array of facial points to be drawn for each frame
    :return: All frames with facial points drawn on them
    """
    edited_frames = []

    if len(frames) != len(facial_points_all):
        logger.warning('Facial points must be provided for every frame')
        return None

    for i in range(0, len(frames)):
        edited_frame = frames[i]
        for point in facial_points_all[i]:
            cv2.circle(edited_frame, (int(point[0]), int(point[1])),
                       1, (0, 255, 0), -1)
        edited_frames.append(edited_frame)

    logger.debug('Facial points drawn successfully')
    return np.array(edited_frames)

def gray_and_resize(frames):
    """
    Turn image to grayscale and resize to UNIVERSAL_RESIZE
    :param frames: All frames of the video to be converted
    """
    edited_frames = []

    for frame in frames:
        frame = resize(frame, width=UNIVERSAL_RESIZE)
        edited_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    return np.array(edited_frames)

def display_video(frames):
    for frame in frames:
        cv2.imshow('Display video', frame)
        if cv2.waitKey(60) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
