import logging
import time

import cv2


logger = logging.getLogger(__name__)

# Data loading configuration
CAM_ADJUSTMENT_TIME = 0.1  # [s]
STOP_VIDEO_KEY = ord('q')


def record_video():
    """
    Record the video using chosen mode
    :return: List of frames (video)
    """
    video_frames = []
    cam = cv2.VideoCapture(0)

    # Start the video
    while True:
        ret, frame = cam.read()
        if not ret:
            logger.warning('Camera is not setup correctly, trying again')
            time.sleep(CAM_ADJUSTMENT_TIME)
            continue

        video_frames.append(frame)

        if logger.isEnabledFor(logging.DEBUG):
            cv2.imshow('Video', frame)
            # Stop the video display by pressing q
            if cv2.waitKey(1) & 0xFF == STOP_VIDEO_KEY:
                break

    return video_frames


def take_photo():
    """
    Take a photo using a chosen method
    :return: Frame of the photo
    """
    cam = cv2.VideoCapture(0)
    time.sleep(CAM_ADJUSTMENT_TIME)
    ret, frame = cam.read()

    if not ret:
        logger.warning('Camera is not setup correctly')

    if logger.isEnabledFor(logging.DEBUG):
        cv2.imshow('Photo', frame)
        cv2.waitKey(0)

    cam.release()
    cv2.destroyAllWindows()
    return frame
