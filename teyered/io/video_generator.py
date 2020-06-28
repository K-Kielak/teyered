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
        self._frames_failed_to_read = 0

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
        logger.info(f'Analysing every {self._frames_to_skip + 1}th frame')
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
            logger.info('Video finished.')
            raise StopIteration('Video is over, cannot read next frame.')

        frame_timespan, frame = self._next_frame_data
        self._read_next_frame()
        return frame_timespan, frame

    def _read_next_frame(self):
        # Skip next frames
        logger.debug(f'Skipping {self._frames_to_skip} frames.')
        for _ in range(self._frames_to_skip):
            if self.is_over():
                logger.debug(f'Reached the end of video during frame skipping.')
                return

            self._vid.read()

        # Read frame
        frame_num = self._vid.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = self._vid.read()

        # If failed to read, try next frames until success or video finishes
        while not ret:
            if self.is_over():
                return

            self._frames_failed_to_read += 1
            logger.warning(f'Frame {frame_num} could not be read.')
            logger.warning(f'Failed to read {self._frames_failed_to_read} '
                           f'out of {frame_num + 1} frames so far.')
            ret, frame = self._vid.read()
            if frame_num == self._vid.get(cv2.CAP_PROP_POS_FRAMES):
                logger.warning(f'Video reading fails to progress further'
                               f'than frame {frame_num} (out of initially '
                               f'planned {self._vid_frames_count}). '
                               f'Finishing reading the video prematurely.')
                self._vid_frames_count = frame_num

            frame_num = self._vid.get(cv2.CAP_PROP_POS_FRAMES)

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