import logging

import cv2


logger = logging.getLogger(__name__)


def load_video(video_file_path, frames_to_skip):
    """
    Load existing video file into the tool
    :param video_file_path: Full path to the video file
    :param frames_to_skip: Every nth frame will be analyzed
    :return: Video frames
    """
    frames = []

    cap = cv2.VideoCapture(video_file_path)

    # Video parameters
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f'\Loading video {video_file_path}:')
    logger.info(f'Width (px): {video_width}')
    logger.info(f'Height (px): {video_height}')
    logger.info(f'FPS: {video_fps}')
    logger.info(f'Frames count: {video_frames_count}')
    logger.info(f'Analysing every {frames_to_skip}th frame')

    counter = -1  # Setting to -1 saves some hassle with lines
    while cap.isOpened():
        # Check if the video is over
        if cap.get(1) == video_frames_count:
            logger.info('Video is over')
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

    logger.info(f'Video loading for {video_file_path} has finished')

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


def save_points(file_path, data):
    """
    Write (x,y) points to the specified file path
    :param file_path: Full path to the file
    :param data: Data in format [(x1, y1), ... ,(xn, yn)]
    """
    logger.info(f'Writing data to file {file_path}')

    with open(file_path, 'w') as f:
        f.write('x,y\n')
        for d in data:
            f.write(f'{d[0]},{d[1]}\n')

    logger.info('Data has been successfully written to file')
