import os

import numpy as np


# Important paths
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCES_DIR = os.path.join(PROJECT_ROOT_DIR, 'resources')

PREDICTOR_FILENAME = 'shape_predictor_68_face_landmarks.dat'
PREDICTOR_FILEPATH = os.path.join(RESOURCES_DIR, PREDICTOR_FILENAME)

FACE_MODEL_FILENAME = 'face_model.txt'
FACE_MODEL_FILEPATH = os.path.join(RESOURCES_DIR, FACE_MODEL_FILENAME)

# Image processing configuration
IMAGE_UPSAMPLE_FACTOR = 1  # Ease facial landmark detection (value from dlib)
UNIVERSAL_RESIZE = 1080 # [px] Photo processing size
ASPECT_RATIO = 720/1080

# Lower and upper boundaries of the facial feature labels
JAW_COORDINATES = (0, 17)
LEFT_EYEBROW_COORDINATES = (17, 22)
RIGHT_EYEBROW_COORDINATES = (22, 27)
NOSE_COORDINATES = (27, 36)
RIGHT_EYE_COORDINATES = (36, 42)
LEFT_EYE_COORDINATES = (42, 48)
MOUTH_COORDINATES = (48, 68)

# Need to know aspect ratio
CAMERA_MATRIX = np.array(
                             [[962.51477715, 0, 509.06946124],
                             [0, 961.84061336, 337.23457898],
                             [0, 0, 1]], dtype = "double"
                         )
DIST_COEFFS = np.array([[-0.0992409, 1.0407034, -0.00665748, -0.01156595, -2.11200394]], dtype="double")

"""
# Approximation for camera parameters
FOCAL_LENGTH = UNIVERSAL_RESIZE
CENTER = (UNIVERSAL_RESIZE / 2, UNIVERSAL_RESIZE*ASPECT_RATIO / 2)  # Need to know aspect ratio
CAMERA_MATRIX = np.array(
                             [[FOCAL_LENGTH, 0, CENTER[0]],
                             [0, FOCAL_LENGTH, CENTER[1]],
                             [0, 0, 1]], dtype = "double"
                         )
DIST_COEFFS = np.zeros((4, 1))
"""