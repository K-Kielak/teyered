import logging
from collections import defaultdict

import cv2
import dlib
from imutils import face_utils, resize

from teyered.config import IMAGE_UPSAMPLE_FACTOR, PREDICTOR_FILEPATH
from teyered.io.image_correction import gray_and_resize


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

    def _track_feature_points_LK(self, previous_frame, new_frame,
                                 previous_points, detected_points):
        """
        Track facial points using Lucas-Kanade optical flow
        :param previous_frame: previous frame numpy array in grayscale resized
        :param new_frame: new frame numpy array in grayscale resized
        :param previous_points: feature points in previous frame
        :param detected_points: detected points in the new frame
        :return: tracked points
        """
        # Get the estimate of the new points
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
             previous_frame_gray, new_frame_gray,
             previous_points.astype(np.float32), None, **self._LK_PARAMS)

        # Get the good points according to the status
        points_status = np.hstack((new_points, status))
        new_points_good = np.array(points_status[points_status[:,2] == 1],
                                   dtype=np.float32)

        # If some points were not tracked, redetect all (todo: redetect only those
        # which are needed based on errors of individual points, for example by
        # using s.d. of all points or somehow else)
        if new_points_good.shape[0] != detected_points.shape[0]:
            return None

        return new_points_good[:,0:2]

    def _detect_facial_points(self, frame):
        """
        Detect facial points of a single face in a frame using dlib
        :param frame: grayscale resized frame
        :return: detected facial features in the frame
        """
        for (i, rect) in enumerate(self._detector(gray, IMAGE_UPSAMPLE_FACTOR)):
            facial_points = self._predictor(gray, rect)
            facial_points = face_utils.shape_to_np(facial_points)
            return facial_points

    def extract_facial_points(self, image):
        """
        Extract facial points from a video using detection and tracking
        :param frames: array of frames (numpy array) to be analysed
        :return: numpy array of all facial points in each frame
        """
        facial_points_all = []

        # Previous frame and points for tracking
        previous_frame = None
        previous_points = None

        images_not_used = frames.shape[0]
        frame_count = 0

        for frame in frames:
            frame = gray_and_resize(frame)
            detected_facial_points = self._detect_facial_points(frame)

            # No facial points detected, skip this frame (maybe track from previous?)
            if not facial_points:
                facial_points_all.append([])
                images_not_used -= 1
                frame_count = 0
                continue

            # (Re)detect at every TRACKING_LENGTHth frame
            if frame_count % TRACKING_LENGTH == 0:
                previous_points = detected_facial_points
                previous_frame = frame
                frame_count = 0
            # Track
            else:
                previous_points = self._track_feature_points_LK(
                    previous_frame, frame, previous_points,
                    detected_facial_points)
                # Tracking is unsuccessful, redetect
                if not previous_points:
                    previous_points = detected_facial_points
                    frame_count = 0
                previous_frame = frame

            frame_count += 1
            facial_points_all.append(previous_points)

        logger.debug('Facial points were successfully extracted from the image')
        return np.array(facial_points_all)
