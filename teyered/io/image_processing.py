import logging

import cv2
from imutils import resize
import numpy as np

from teyered.config import UNIVERSAL_RESIZE, RED_COLOR, GREEN_COLOR, \
    BLUE_COLOR, WHITE_COLOR, CAMERA_MATRIX, DIST_COEFFS


logger = logging.getLogger(__name__)


def draw_pose(frames, facial_points_all, r_vectors_all, t_vectors_all,
              camera_matrix=CAMERA_MATRIX, dist_coeffs=DIST_COEFFS):
    """
    Draw 3D pose estimation using basis vectors projected onto image
    coordinate system
    :param frames: Video frames scaled to other parameters
    :param facial_points_all: Facial points in each frame
    :param r_vectors_all: Rotation vectors in each frame
    :param t_vectors_all: Translation vectors in each frame
    :param camera_matrix: Calibrated camera matrix
    :param dist_coeffs: Calibrated distortion coefficients
    :return: All frames with pose drawn on them
    """
    if frames.shape[0] != facial_points_all.shape[0]:
        raise ValueError(
            'facial_points_all and frames array length must be the same')
    if frames.shape[0] != r_vectors_all.shape[0]:
        raise ValueError(
            'r_vectors_all and frames array length must be the same')
    if frames.shape[0] != t_vectors_all.shape[0]:
        raise ValueError(
            't_vectors_all and frames array length must be the same')
    if frames.size < 1:
        raise ValueError('Must provide at least one frame')

    edited_frames = []

    for i, frame in enumerate(frames):
        edited_frame = frame

        # No facial points identified for the frame
        if facial_points_all[i] is None:
            edited_frames.append(edited_frame)
            continue

        # Project 3D points onto 2D image plane based on the coordinate system
        # described in README, but inverse y-axis and z-axis, so that
        # it is more human friendly

        rot_x, _ = cv2.projectPoints(np.array([(1.0, 0.0, 0.0)]),
                                     r_vectors_all[i], t_vectors_all[i],
                                     camera_matrix, dist_coeffs)
        rot_y, _ = cv2.projectPoints(np.array([(0.0, -1.0, 0.0)]),
                                     r_vectors_all[i], t_vectors_all[i],
                                     camera_matrix, dist_coeffs)
        rot_z, _ = cv2.projectPoints(np.array([(0.0, 0.0, -1.0)]),
                                     r_vectors_all[i], t_vectors_all[i],
                                     camera_matrix, dist_coeffs)
        rot_origin, _ = cv2.projectPoints(np.array([(0.0, 0.0, 0.0)]),
                                          r_vectors_all[i], t_vectors_all[i],
                                          camera_matrix, dist_coeffs)

        p_x = (int(rot_x[0][0][0]), int(rot_x[0][0][1]))
        p_y = (int(rot_y[0][0][0]), int(rot_y[0][0][1]))
        p_z = (int(rot_z[0][0][0]), int(rot_z[0][0][1]))
        p_origin = (int(rot_origin[0][0][0]), int(rot_origin[0][0][1]))

        cv2.line(edited_frame, p_origin, p_x, BLUE_COLOR, 2)
        cv2.line(edited_frame, p_origin, p_y, GREEN_COLOR, 2)
        cv2.line(edited_frame, p_origin, p_z, RED_COLOR, 2)

        edited_frames.append(edited_frame)

    logger.debug('Pose vectors drawn successfully')
    return np.array(edited_frames)


def draw_facial_points(frames, facial_points_all, color=GREEN_COLOR):
    """
    :param frames: Video frames scaled to facial points
    :param facial_points_all: Facial points to be drawn for each frame
    :param color: Color of the facial points
    :return: All frames with facial points drawn on them
    """
    if frames.shape[0] != facial_points_all.shape[0]:
        raise ValueError(
            'facial_points_all and frames array length must be the same')
    if frames.size < 1:
        raise ValueError('Must provide at least one frame')

    edited_frames = []

    for i, frame in enumerate(frames):
        edited_frame = frame

        if facial_points_all[i] is not None:
            for point in facial_points_all[i]:
                cv2.circle(edited_frame, (int(point[0]), int(point[1])),
                        1, color, -1)

        edited_frames.append(edited_frame)

    logger.debug('Facial points drawn successfully')
    return np.array(edited_frames)


def draw_projected_points(frames, projected_points_all, color=RED_COLOR):
    """
    :param frames: Video frames scaled to facial points
    :param projected_points_all: Projected points to be drawn for each frame
    :param color: Color of the projected points
    :return: All frames with projected points drawn on them
    """
    if frames.shape[0] != projected_points_all.shape[0]:
        raise ValueError(
            'projected_points and frames array length must be the same')
    if frames.size < 1:
        raise ValueError('Must provide at least one frame')

    edited_frames = []

    for i, frame in enumerate(frames):
        edited_frame = frame

        if projected_points_all[i] is not None:
            for point in projected_points_all[i]:
                cv2.circle(edited_frame, (int(point[0][0]), int(point[0][1])), 1, color, -1)

        edited_frames.append(edited_frame)

    logger.debug('Projected points drawn successfully')
    return np.array(edited_frames)


def write_angles(frames, angles_all, color=WHITE_COLOR):
    """
    Write the three angles (yaw, pitch, roll) on top of the video
    :param frames: Video frames scaled to facial points
    :param angles_all: Angles to be drawn for each frame
    :param color: Color of the angles
    :return: All frames with angles written on them
    """
    if frames.shape[0] != angles_all.shape[0]:
        raise ValueError('angles_all and frames array length must be the same')
    if frames.size < 1:
        raise ValueError('Must provide at least one frame')

    edited_frames = []

    for i, frame in enumerate(frames):
        edited_frame = frame

        if angles_all[i] is None:
            edited_frames.append(edited_frame)
            continue

        angles_all[i] = np.around(angles_all[i], decimals=1)

        angle_text_yaw = f'yaw: {angles_all[i][0][0]}'
        angle_text_pitch = f'pitch: {angles_all[i][1][0]}'
        angle_text_roll = f'roll: {angles_all[i][2][0]}'

        cv2.putText(edited_frame, angle_text_yaw, (10, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(edited_frame, angle_text_pitch, (10, 275),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(edited_frame, angle_text_roll, (10, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        edited_frames.append(edited_frame)

    logger.debug('Angles written successfully')
    return np.array(edited_frames)


def write_closedness(frames, closedness_left_all, closedness_right_all, color=WHITE_COLOR):
    """
    Write closedness for both eyes on top of the video
    :param frames: Video frames scaled to facial points
    :param closedness_left_all: Closedness of left eye for all frames
    :param closedness_right_all: Closedness of right eye for all frames 
    :param color: Color of the text
    :return: All frames with angles text writtten on them
    """
    if frames.shape[0] != closedness_left_all.shape[0]:
        raise ValueError('closedness_left_all and frames array length must be the same')
    if frames.shape[0] != closedness_right_all.shape[0]:
        raise ValueError('closedness_right_all and frames array length must be the same')
    if frames.size < 1:
        raise ValueError('Must provide at least one frame')
    
    edited_frames = []

    for i, frame in enumerate(frames):
        edited_frame = frame

        if closedness_left_all[i] is None and closedness_right_all[i] is None:
            edited_frames.append(edited_frame)
            continue

        closedness_left_text = f'closedness left: {int(closedness_left_all[i]*100)}%'
        closedness_right_text = f'closedness right: {int(closedness_right_all[i]*100)}%'

        cv2.putText(edited_frame, closedness_left_text, (10, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(edited_frame, closedness_right_text, (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        edited_frames.append(edited_frame)

    logger.debug('Closedness data written successfully')
    return np.array(edited_frames)


def gray_video(frames):
    """
    :param frames:
    :return: Video in grayscale
    """
    if frames.size < 1:
        raise ValueError('Must provide at least one frame')

    edited_frames = []

    for i, frame in enumerate(frames):
        edited_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    logger.debug('Video converted to grayscale successfully')
    return np.array(edited_frames)


def resize_video(frames, width: int = UNIVERSAL_RESIZE):
    """
    :param frames:
    :return: Resized video
    """
    if frames.size < 1:
        raise ValueError('Must provide at least one frame')

    edited_frames = []

    for i, frame in enumerate(frames):
        edited_frames.append(resize(frame, width))

    logger.debug('Video resized successfully')
    return np.array(edited_frames)

def display_video(frames):
    """
    :param frames:
    """
    if frames.size < 2:
        raise ValueError('Must provde at least one frame')

    for frame in frames:
        cv2.imshow('Display video', frame)

        if cv2.waitKey(60) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def display_image(frame):
    """
    :param frame:
    """
    cv2.imshow('Display video', frame)

    return not (cv2.waitKey(60) & 0xFF == ord('q'))