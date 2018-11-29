import logging
import os
import time

import cv2


# Absolute path to the file directory
FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)
LOG_FILE_NAME = 'data_recording.log'
LOG_DIRECTORY = os.path.abspath(
    os.path.join(FILE_DIRECTORY, *[os.pardir, 'logs', LOG_FILE_NAME])
)
logger_formatter = logging.Formatter('%(asctime)s : %(name)s : %(message)s')
file_handler = logging.FileHandler(LOG_DIRECTORY)
file_handler.setFormatter(logger_formatter)
logger.addHandler(file_handler)

# Time camera takes to adjust exposure and iso after opening
CAM_ADJUSTMENT_TIME = 0.1  # [s]

def record_video():
    """
    Record the video using chosen mode
    :return: List of frames (video)
    """
    video_frames = []
    cam = cv2.VideoCapture(0)

    # Start the video
    while(True):
        ret, frame = cam.read()
        if not ret:
            log.warning('Camera is not setup correctly, trying again')
            time.sleep(0.1)
            continue

        video_frames.append(frame)

        if logger.isEnabledFor(logging.DEBUG):
            cv2.imshow("Video", frame)
            # Stop the video display by pressing q
            if cv2.waitKey(1) & 0xFF == ord('q'):
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
        log.warning('Camera is not setup correctly')

    if logger.isEnabledFor(logging.DEBUG):
        cv2.imshow("Photo", frame)
        cv2.waitKey(0)

    cam.release()
    cv2.destroyAllWindows()
    return frame
