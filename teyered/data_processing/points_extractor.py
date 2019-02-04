import logging
from collections import defaultdict

import cv2
import dlib
from imutils import face_utils, resize

from teyered.config import IMAGE_UPSAMPLE_FACTOR, PREDICTOR_FILEPATH, \
    UNIVERSAL_RESIZE


logger = logging.getLogger(__name__)


class FacialPointsExtractor:

    def __init__(self):
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(PREDICTOR_FILEPATH)

    def extract_facial_points(self, image):
        """
        Extract facial features and corresponding points from the input image
        :param image: Image frame numpy array to be processed
        :return: Facial features and corresponding points of the image
        """
        facial_points = defaultdict(list)
        image = resize(image, width=UNIVERSAL_RESIZE)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces in the grayscale image
        for i, rect in enumerate(self._detector(gray, IMAGE_UPSAMPLE_FACTOR)):
            shape = self._predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            for name, (start, end) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                facial_points[name].extend(shape[start:end])

        logger.debug('Facial points were successfully extracted from the image')
        return facial_points
