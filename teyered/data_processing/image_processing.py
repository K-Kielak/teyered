import logging

import cv2
import numpy as np
from imutils import resize

from teyered.config import UNIVERSAL_RESIZE


logger = logging.getLogger(__name__)


def gray_video(frames):
    """
    Convert all frames into grayscale
    :param frames: All frames
    :return: Video in grayscale
    """
    if frames.size < 1:
        raise ValueError('Must provide at least one frame')

    edited_frames = []

    for frame in frames:
        edited_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    logger.debug('Video converted to grayscale successfully')
    return np.array(edited_frames)


def resize_video(frames, width: int = UNIVERSAL_RESIZE):
    """
    Resize all frames to the selected width
    :param frames: All frames
    :param width: Selected width
    :return: Resized video
    """
    if frames.size < 1:
        raise ValueError('Must provide at least one frame')

    edited_frames = []

    for frame in frames:
        edited_frames.append(resize(frame, width))

    logger.debug('Video resized successfully')
    return np.array(edited_frames)
