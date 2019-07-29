import logging
import os

import cv2


logger = logging.getLogger(__name__)


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

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_next_frame()

    def get_next_frame(self):
        if not self.is_open():
            raise IOError('Cannot process the video, file is not open.')

        if self.is_over():
            raise StopIteration('Video is over, cannot read next frame.')

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