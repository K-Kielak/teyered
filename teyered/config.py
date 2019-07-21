import os

import numpy as np


# Important paths
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCES_DIR = os.path.join(PROJECT_ROOT_DIR, 'resources')
PRERECORDED_VIDEO_DIR = os.path.join(PROJECT_ROOT_DIR, 'test_footage')
REPORTS_DIR = os.path.join(PROJECT_ROOT_DIR, 'reports')

# Resources paths
PREDICTOR_FILENAME = 'shape_predictor_68_face_landmarks.dat'
PREDICTOR_FILEPATH = os.path.join(RESOURCES_DIR, PREDICTOR_FILENAME)

FACE_MODEL_FILENAME = 'face_model.txt'
FACE_MODEL_FILEPATH = os.path.join(RESOURCES_DIR, FACE_MODEL_FILENAME)

FACE_GROUND_TRUTH_FILENAME = 'ground_truth.jpg'
FACE_GROUND_TRUTH_FILEPATH = os.path.join(RESOURCES_DIR, FACE_GROUND_TRUTH_FILENAME)

# Prerecorded videos paths
PRERECORDED_VIDEO_FILENAME = 'video.mov'
PRERECORDED_VIDEO_FILEPATH = os.path.join(PRERECORDED_VIDEO_DIR, PRERECORDED_VIDEO_FILENAME)

# Reports paths
ERROR_DATA_FILENAME = 'error_data.csv'
ERROR_DATA_FILEPATH = os.path.join(REPORTS_DIR, ERROR_DATA_FILENAME)

FP_DATA_FILENAME = 'fp_data.csv'
FP_DATA_FILEPATH = os.path.join(REPORTS_DIR, FP_DATA_FILENAME)

EYE_DATA_FILENAME = 'eye_data.csv'
EYE_DATA_FILEPATH = os.path.join(REPORTS_DIR, EYE_DATA_FILENAME)

POSE_DATA_FILENAME = 'pose_data.csv'
POSE_DATA_FILEPATH = os.path.join(REPORTS_DIR, POSE_DATA_FILENAME)

# Image processing configuration
IMAGE_UPSAMPLE_FACTOR = 1  # Ease facial landmark detection (value from dlib)
UNIVERSAL_RESIZE = 500 # [px] Photo processing size
ASPECT_RATIO = 720/1080

# Lower and upper boundaries of the facial feature labels
JAW_COORDINATES = (0, 17)
LEFT_EYEBROW_COORDINATES = (17, 22)
RIGHT_EYEBROW_COORDINATES = (22, 27)
NOSE_COORDINATES = (27, 36)
RIGHT_EYE_COORDINATES = (36, 42)
LEFT_EYE_COORDINATES = (42, 48)
MOUTH_COORDINATES = (48, 68)

# How many consecutive frames to track before redetection
TRACKING_LENGTH = 5

# Analyse every FRAME_TO_ANALYSEth frame. = 1, then analyse every frame. = 2, then analyse every second frame etc.
FRAME_TO_ANALYSE = 1

# How many seconds of footage to analyse in a single iteration
BATCH_SIZE = 10 # seconds

"""
# Leo and Karl's camera parameters (calibrated)
CAMERA_MATRIX = np.array(
                             [[962.51477715, 0, 509.06946124],
                             [0, 961.84061336, 337.23457898],
                             [0, 0, 1]], dtype = "double"
                         )
DIST_COEFFS = np.array([[-0.0992409, 1.0407034, -0.00665748, -0.01156595, -2.11200394]], dtype="double")
"""

# Approximation for any camera parameters
FOCAL_LENGTH = UNIVERSAL_RESIZE
CENTER = (UNIVERSAL_RESIZE / 2, UNIVERSAL_RESIZE*ASPECT_RATIO / 2)  # Need to know aspect ratio
CAMERA_MATRIX = np.array(
                             [[FOCAL_LENGTH, 0, CENTER[0]],
                             [0, FOCAL_LENGTH, CENTER[1]],
                             [0, 0, 1]], dtype = "double"
                         )
DIST_COEFFS = np.zeros((4, 1))

# Colours (hex inversed, #0000ff is blue, but here red so read from right to left)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)
WHITE_COLOR = (255, 255, 255)
BLACK_COLOR = (0, 0, 0)

# 3D colours for pyqtgraph and opengl (3 RGB, transparancy)
RED_COLOR_3D = (1.0, 0.0, 0.0, 1.0)
GREEN_COLOR_3D = (0.0, 1.0, 0.0, 1.0)
BLUE_COLOR_3D = (0.0, 0.0, 1.0, 1.0)
YELLOW_COLOR_3D = (1.0, 1.0, 0.0, 1.0)
