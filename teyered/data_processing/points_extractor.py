import logging

import cv2
import dlib
import numpy as np
from imutils import face_utils

from teyered.config import IMAGE_UPSAMPLE_FACTOR, PREDICTOR_FILEPATH
from teyered.config import TRACKING_LENGTH


logger = logging.getLogger(__name__)


class FacialPointsExtractor:

    def __init__(self):
        # dlib detection and prediction
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(PREDICTOR_FILEPATH)

        # LK optical flow tracking parameters
        self._LK_PARAMS = dict(winSize=(15, 15), maxLevel=3,
                               criteria=(cv2.TERM_CRITERIA_EPS |
                                         cv2.TERM_CRITERIA_COUNT,
                                         10, 0.03))

        # Previous batch information
        self._previous_frame = None
        self._previous_points = None
        self._frame_count = 0

    def reset(self):
        self._previous_frame = None
        self._previous_points = None
        self._frame_count = 0

    def get_previous_frame(self):
        return self._previous_frame

    def get_previous_points(self):
        return self._previous_points

    def get_frame_count(self):
        return self._frame_count

    def _track_facial_points_LK(self, new_frame, detected_points):
        """
        Track facial points using Lucas-Kanade optical flow
        TODO Redetect individual points based on their errors
        TODO Try other tracking methods
        :param new_frame: Current frame in grayscale scaled
        :param detected_points: Detected points in new frame for relative
        error estimation
        :return: np.ndarray of tracked points or None if no points are tracked
        """
        # Get the estimate of the new points
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self._previous_frame, new_frame,
            self._previous_points.astype(np.float32), None, **self._LK_PARAMS
        )

        # Get the points which were tracked
        points_status = np.hstack((new_points, status))
        new_points_good = np.array(points_status[points_status[:, 2] == 1],
                                   dtype=np.float32)

        # If some points were not tracked, redetect all
        if new_points_good.shape[0] != detected_points.shape[0]:
            return None

        return new_points_good[:, 0:2]

    def _detect_facial_points(self, frame):
        """
        Detect facial points of the first detected face in a frame using dlib
        TODO Change the detection to our CNN
        TODO Deal with multiple faces being discovered
        :param frame: Frame grayscale scaled
        :return: np.ndarray of detected facial features in the frame scaled
        to frame size or None if not detected
        """
        # Rectangles of multiple faces
        rect = self._detector(frame, IMAGE_UPSAMPLE_FACTOR)

        # Only use the first face image
        if len(rect) > 0:
            facial_points = self._predictor(frame, rect[0])
            facial_points = face_utils.shape_to_np(facial_points)
            return facial_points
        else:
            return None

    def extract_facial_points(self, frames):
        """
        Extract facial points from a video sequence combining detection and
        tracking
        TODO Introduce filtering (like Kalman)
        :param frames: All frames to be analysed
        :return: np.ndarray of all facial points in each frame
        """
        if frames.shape[0] < 1:
            raise ValueError('At least one frame is required')

        # Information to be returned
        facial_points_all = []

        for frame in frames:
            # Detect facial points as this will be needed in any case
            detected_facial_points = self._detect_facial_points(frame)

            # No facial points detected, skip this frame and detect in the
            # next iteration
            if detected_facial_points is None:
                facial_points_all.append(None)
                self._previous_points = None
                self._previous_frame = None
                self._frame_count = 0
                continue

            # (Re)detect at every TRACKING_LENGTHth frame
            if self._frame_count % TRACKING_LENGTH == 0:
                self._previous_points = np.copy(detected_facial_points)
                self._previous_frame = np.copy(frame)
                self._frame_count = 1
            # Track
            else:
                self._previous_points = self._track_facial_points_LK(
                    frame, detected_facial_points
                )
                self._previous_frame = np.copy(frame)
                self._frame_count += 1

                # Tracking is unsuccessful, redetect
                if self._previous_points is None:
                    self._previous_points = np.copy(detected_facial_points)
                    self._frame_count = 1

            facial_points_all.append(np.copy(self._previous_points))

        logger.debug('Facial points were successfully extracted from the '
                     'video batch')
        return np.array(facial_points_all)
