import cv2
from imutils import face_utils
import numpy as np
import imutils
import dlib
import sys
import os

# Pictures are resized and points are stored at this width
RESIZE_WIDTH = 500  # [px]

def extract_facial_points(image):
    """
    Extract facial features and corresponding points from the input image
    :param image: Image frame numpy array to be processed
    :return: Facial features and corresponding points of the image
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(os.path.join(os.path.dirname(os.path.abspath(__file__)) , 'shape_predictor_68_face_landmarks.dat'))

    # Dictionary for the facial features and corresponding points
    facial_points = { "left_eye" : [], "right_eye" : [] }

    # Resize the image to 500x500 and transform into a gray picture
    image = imutils.resize(image, width=RESIZE_WIDTH)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    for (i, rect) in enumerate(detector(gray, 1)):

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():

            for (x, y) in shape[i:j]:
                if (i, j) == (36,42):  # Right eye
                    facial_points["right_eye"].append((x,y))
                elif (i, j) == (42,48):  # Left eye
                    facial_points["left_eye"].append((x,y))
    
    return facial_points