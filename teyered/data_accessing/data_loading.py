import logging
import os

import cv2


# Absolute path to the file directory
FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)
LOG_FILE_NAME = 'data_loading.log'
LOG_DIRECTORY = os.path.abspath(
    os.path.join(FILE_DIRECTORY, *[os.pardir, 'logs', LOG_FILE_NAME])
)
logger_formatter = logging.Formatter('%(asctime)s : %(name)s : %(message)s')
file_handler = logging.FileHandler(LOG_DIRECTORY)
file_handler.setFormatter(logger_formatter)
logger.addHandler(file_handler)


def load_video(video_file_path, frames_to_skip):
    """
    Load existing video file into the tool
    :param video_file_path: Full path to the video file
    :param frames_to_skip: Every nth frame will be analyzed
    :return: Video frames
    """
    video_frames = []

    cap = cv2.VideoCapture(video_file_path)

    # Video parameters
    video_width = int(cap.get(cv2.CV_CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CV_CAP_PROP_FRAME_HEIGHT))
    video_fps = int(cap.get(cv2.CV_CAP_PROP_FPS))
    video_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.debug(f'\Loading video {video_file_path}:')
    logger.debug(f'Width (px): {video_width}')
    logger.debug(f'Height (px): {video_height}')
    logger.debug(f'FPS: {video_fps}')
    logger.debug(f'Frames count: {video_frames_count}')
    logger.debug(f'Analysing every {frames_to_skip}th frame')

    counter = -1  # Setting to -1 saves some hassle with lines
    while(cap.isOpened()):
        # Check if the video is over
        if cap.get(1) == video_frames_count:
            logger.debug('Video is over')
            break

        ret, frame = cap.read()
        counter += 1

        if not ret:
            logger.warning(f'Frame {counter} could not be read')
            continue

        # Showing only every n-th frame
        if (counter % frames_to_skip) != 0:
            continue

        frames.append(frame)

    logger.debug(f'Video loading for {video_file_path} has finished')

    cap.release()
    cv2.destroyAllWindows()
    return frames

def load_image(image_file_path):
    """
    Load existing image from a specified file path into the tool
    :param image_file_path: Full file path to the image
    :return: Frame of the image
    """
    frame = cv2.imread(image_file_path)

    if logger.isEnabledFor(logging.DEBUG):
        cv2.imshow(image_file_path, frame)
        cv2.waitKey(0)

    return frame
