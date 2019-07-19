import logging
import os

import cv2
import numpy as np

from teyered.config import UNIVERSAL_RESIZE, ERROR_DATA_FILEPATH, FP_DATA_FILEPATH, EYE_DATA_FILEPATH, POSE_DATA_FILEPATH
from teyered.io.image_processing import display_image, display_video


logger = logging.getLogger(__name__)


"""
Methods and classes for solving memory problem - use with large files
"""


class VideoGenerator:

    def __init__(self, video_path, frames_to_skip):
        self._vid_path = video_path
        self._frames_to_skip = frames_to_skip
        self._vid = None
        self._vid_fps = None
        self._vid_frames_count = None
        self._next_frame_data = None

    def __enter__(self):
        self.open()
        return self

    def open(self):
        logger.info(f'Loading video {self._vid_path}:')
        if not os.path.isfile(self._vid_path):
            raise AttributeError(f'File like {self._vid_path} does not exist.')

        self._vid = cv2.VideoCapture(self._vid_path)
        self._vid_fps = int(self._vid.get(cv2.CAP_PROP_FPS))
        self._vid_frames_count = int(self._vid.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f'Width (px): {self._vid.get(cv2.CAP_PROP_FRAME_WIDTH)}')
        logger.info(f'Height (px): {self._vid.get(cv2.CAP_PROP_FRAME_HEIGHT)}')
        logger.info(f'FPS: {self._vid_fps}')
        logger.info(f'Frames count: {self._vid_frames_count}')
        logger.info(f'Analysing every {self._frames_to_skip}th frame')
        self._read_next_frame()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if not self.is_open():
            raise IOError('Cannot close the video, '
                          'it is not open in the first place.')
        logger.info(f'Closing video {self._vid_path}:')
        self._vid.release()

    def get_next_frame(self):
        if not self.is_open():
            raise IOError('Cannot process the video, file is not open.')

        if self.is_over():
            raise IOError('Video is over, cannot read next frame.')

        frame_timespan, frame = self._next_frame_data
        self._read_next_frame()
        return frame_timespan, frame

    def _read_next_frame(self):
        # Skip frames
        frame_num = self._vid.get(cv2.CAP_PROP_POS_FRAMES)
        while not self.is_over() and frame_num % self._frames_to_skip != 0:
            self._vid.read()
            frame_num = self._vid.get(cv2.CAP_PROP_POS_FRAMES)

        if self.is_over():
            logger.info('Video finished')
            return

        # Read frame to memory
        ret, frame = self._vid.read()
        if not ret:
            logger.warning(f'Frame {frame_num} could not be read.')
            self._read_next_frame()
            return

        logger.debug(f'Loading frame no:. {frame_num}')
        frame_timespan = frame_num / self._vid_fps  # [s]
        self._next_frame_data = (frame_timespan, frame)

    def is_open(self):
        return self._vid is not None and self._vid.isOpened()

    def is_over(self):
        if not self.is_open():
            raise IOError('Cannot process the video, file is not open.')

        curr_frame_num = self._vid.get(cv2.CAP_PROP_POS_FRAMES)
        return curr_frame_num == self._vid_frames_count


"""
Methods and classes for writing directly to app memory - avoid for large files
"""


def load_video(video_file_path):
    """
    Load the whole video into memory
    :param video_file_path: Relative path to the video
    :return: Frames of the video
    """
    frames = []

    cap = cv2.VideoCapture(video_file_path)
    video_frames_count = extract_video_information(cap)['frames_count']

    while cap.isOpened():
        if cap.get(1) == video_frames_count:
            break

        ret, frame = cap.read()
        frames.append(frame)

    cap.release()

    if logger.isEnabledFor(logging.DEBUG):
        display_video(frame)

    cv2.destroyAllWindows()
    
    logger.info(f'Video successfully loaded from {video_file_path}')
    return np.array(frames)

def load_image(image_file_path):
    """
    Load existing image into memory
    :param image_file_path: Relative file path to the image
    :return: Image
    """
    frame = cv2.imread(image_file_path)

    if logger.isEnabledFor(logging.DEBUG):
        display_image(frame)

    logger.info(f'Image successfully loaded from {image_file_path}')
    return frame

def save_video(frames, name, format, folder = ''):
    """
    Save video from memory to a file
    :param frames: Frames in memory
    :param name: Name without the extension
    :param format: Format of the file, either mp4 or avi
    :param folder: Absolute path to the folder (default - working directory)
    """
    file_path = os.path.join(folder, f'{name}.{format}')

    if format == 'mp4':
        out = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'MP4V'),
                              24, (frames[0].shape[1],frames[0].shape[0]))
    elif format == 'avi':
        out = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc('M','J','P','G'),
                              24, (frames[0].shape[1],frames[0].shape[0]))
    else:
        raise ValueError('Format of the video can only be mp4 or avi')

    for frame in frames:
        out.write(frame)

    out.release()
    logger.info(f'Video successfully saved as {file_path}')

def save_image(frame, name, format, folder = ''):
    """
    Save image from memory to a file in a working directory
    :param frame: Image in memory
    :param name: Name without the extension
    :param format: Format of the image, either jpg or png
    :param folder: Absolute path to the folder (default - working directory)
    """
    file_path = os.path.join(folder, f'{name}.{format}')

    if not format == 'jpg' or not format == 'png':
        raise ValueError('Format of the image can only be jpg or png')
    
    cv2.imwrite(file_path, frame)
    logger.info(f'Image successfully saved as {file_path}')

def extract_video_information(video):
    """
    :param video: cv2.VideoCapture object
    :return: Dictionary of video parameters
    """
    video_params = {}

    video_params['width'] = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_params['height'] = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    video_params['fps'] = int(video.get(cv2.CAP_PROP_FPS))
    video_params['frames_count'] = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f'Width (px): {video_params["width"]}')
    logger.info(f'Height (px): {video_params["height"]}')
    logger.info(f'FPS: {video_params["fps"]}')
    logger.info(f'Frames count: {video_params["frames_count"]}')

    logger.info('Video parameters were successfully fetched')
    return video_params


"""
Methods for saving specific information to csv files
"""


def initialize_reports(error_path = ERROR_DATA_FILEPATH, pose_path = POSE_DATA_FILEPATH, fp_path = FP_DATA_FILEPATH, eye_path = EYE_DATA_FILEPATH):
    """
    Initialize csv files and their headers for further data storage
    :param error_path: Relative path to save reprojection error data
    :param pose_path: Relative path to save pose data
    :param fp_path: Relative path to save facial points data
    """
    # Reprojection error (frame, error)
    with open(error_path, 'w') as f:
        initialization_string = 'frame,error\n'
        f.write(initialization_string)
    logger.info(f'Error data csv file was successfully initialized at {error_path}')
        
    # Pose (frame, yaw, pitch, roll)
    with open(pose_path, 'w') as f:
        initialization_string = 'frame,yaw,pitch,roll\n'
        f.write(initialization_string)
    logger.info(f'Pose data csv file was successfully initialized at {pose_path}')
    
    # Facial points (frame, 1x, 1y, 2x, 2y, ..., 68x, 68y)
    with open(fp_path, 'w') as f:
        initialization_string = 'frame,'
        
        for i in range(0,68):
            initialization_string += f'{i+1}x,{i+1}y,'
        initialization_string  = initialization_string[:-1] + '\n'
        
        f.write(f'{initialization_string}')
    logger.info(f'Facial points data csv file was successfully initialized at {fp_path}')

    # Eye closedness (frame, left eye closedness, right eye closedness)
    with open(eye_path, 'w') as f:
        initialization_string = 'frame,left,right\n'
        f.write(initialization_string)
    logger.info(f'Eye closedness data csv file was successfully initialized at {eye_path}')

def save_error_report(frames, data, file_path = ERROR_DATA_FILEPATH):
    """
    Save reprojection error to the initialized csv file
    :param frames: List of frames numbers which are to be saved 
    :param data: Corresponding reprojection error data
    :param file_path: Initialized path file
    """
    if frames.shape[0] != data.shape[0]:
        raise ValueError('frames and data array lengths must be the same')

    with open(file_path, 'a') as f:
        for i, d in enumerate(data):
            d_string = f'{frames[i]},{d[1]}\n'            
            f.write(d_string)

    logger.info(f'Reprojection error data has been successfully written to {file_path}')

def save_pose_report(frames, data, file_path = POSE_DATA_FILEPATH):
    """
    Save pose data to the initialized csv file
    :param frames: List of frames numbers which are to be saved 
    :param data: Corresponding pose data
    :param file_path: Initialized path file
    """
    if frames.shape[0] != data.shape[0]:
        raise ValueError('frames and data array lengths must be the same')

    with open(file_path, 'a') as f:
        for i, d in enumerate(data):
            d_string = f'{frames[i]},'

            for z in d:
                d_string += f'{z[0]},'
            d_string  = d_string[:-1] + '\n'
            
            f.write(d_string)

    logger.info(f'Pose data has been successfully written to {file_path}')

def save_facial_points_report(frames, data, file_path = FP_DATA_FILEPATH):
    """
    Save facial points data to the initialized csv file
    :param frames: List of frames numbers which are to be saved 
    :param data: Corresponding facial points data
    :param file_path: Initialized path file
    """
    if frames.shape[0] != data.shape[0]:
        raise ValueError('frames and data array lengths must be the same')

    with open(file_path, 'a') as f:
        # Time steps
        for i, d in enumerate(data):
            d_string = f'{frames[i]},'

            # Points
            for z in d:
                d_string += f'{z[0]},{z[1]},' # x,y coordinates
            d_string  = d_string[:-1] + '\n'
                       
            f.write(d_string)
    
    logger.info(f'Facial points data has been successfully written to {file_path}')

def save_eye_closedness_report(frames, data_left, data_right, file_path = EYE_DATA_FILEPATH):
    """
    Save eye closedness data to the initialized csv file
    :param frames: List of frames numbers which are to be saved 
    :param data_left: Corresponding left eye closedness data
    :param data_right: Corresponding right eye closedness data
    :param file_path: Initialized path file
    """
    if frames.shape[0] != data_left.shape[0]:
        raise ValueError('frames and data_left array lengths must be the same')
    if frames.shape[0] != data_right.shape[0]:
        raise ValueError('frames and data_left array lengths must be the same')

    with open(file_path, 'a') as f:
        for i in range(0, data_left.shape[0]):
            d_string = f'{frames[i]},{data_left[i],{data_right[i]}}\n'            
            f.write(d_string)

    logger.info(f'Eye closedness data has been successfully written to {file_path}')