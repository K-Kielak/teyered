import logging

import cv2
import dlib
from imutils import face_utils, resize
import numpy as np

from teyered.config import IMAGE_UPSAMPLE_FACTOR, PREDICTOR_FILEPATH


logger = logging.getLogger(__name__)

# How many consecutive frames to track before redetection
TRACKING_LENGTH = 5


class FacialPointsExtractor:

    def __init__(self):
        # dlib detection and prediction
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(PREDICTOR_FILEPATH)

        # LK optical flow tracking parameters
        self._LK_PARAMS = dict(winSize  = (15,15),
                               maxLevel = 3,
                               criteria = (cv2.TERM_CRITERIA_EPS | \
                                           cv2.TERM_CRITERIA_COUNT,
                                           10, 0.03))

    def _track_facial_points_LK(self, previous_frame, new_frame,
                                previous_points, detected_points):
        """
        Track facial points using Lucas-Kanade optical flow
        :param previous_frame: Previous frame numpy array in grayscale scaled
        :param new_frame: Current frame numpy array in grayscale scaled
        :param previous_points: Feature points in previous frame
        :param detected_points: Detected points in the new frame for relative
        error estimation
        :return: Tracked points
        """
        # Get the estimate of the new points
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
             previous_frame, new_frame, previous_points.astype(np.float32),
             None, **self._LK_PARAMS)

        # Get the points which were tracked
        points_status = np.hstack((new_points, status))
        new_points_good = np.array(points_status[points_status[:,2] == 1],
                                   dtype=np.float32)

        # If some points were not tracked, redetect all (idea: redetect only those
        # which are needed based on errors of individual points, for example by
        # pure pixel distance from detected points)
        if new_points_good.shape[0] != detected_points.shape[0]:
            return None

        return new_points_good[:,0:2]

    def _detect_facial_points(self, frame):
        """
        Detect facial points of the first detected face in a frame using dlib
        :param frame: Frame grayscale scaled
        :return: Detected facial features in the frame
        """
        for (i, rect) in enumerate(self._detector(frame, IMAGE_UPSAMPLE_FACTOR)):
            facial_points = self._predictor(frame, rect)
            facial_points = face_utils.shape_to_np(facial_points)
            return facial_points

    def extract_facial_points(self, frames):
        """
        Extract facial points from a video sequence using detection and tracking
        :param frames: All frames (array of numpy arrays) to be analysed
        :return: List of all facial points in each frame
        """
        if len(frames) < 2:
            logger.warning('At least two frames are required in the sequence')
            return []

        facial_points_all = []

        previous_frame = None
        previous_points = None

        frame_count = 0
        for frame in frames:
            detected_facial_points = self._detect_facial_points(frame)

            # No facial points detected, skip this frame (or track from previous?)
            if detected_facial_points.shape == (0,):
                facial_points_all.append([])
                frame_count = 0
                continue

            # (Re)detect at every TRACKING_LENGTHth frame
            if frame_count % TRACKING_LENGTH == 0:
                previous_points = detected_facial_points
                previous_frame = frame
                frame_count = 0
            # Track
            else:
                previous_points = self._track_facial_points_LK(
                    previous_frame, frame, previous_points,
                    detected_facial_points)
                # Tracking is unsuccessful, redetect
                if previous_points is None:
                    previous_points = detected_facial_points
                    frame_count = 0
                previous_frame = frame

            frame_count += 1
            facial_points_all.append(previous_points)

        logger.debug('Facial points were successfully extracted from the image')
        return facial_points_all
