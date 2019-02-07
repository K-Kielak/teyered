import os


# Important paths
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCES_DIR = os.path.join(PROJECT_ROOT_DIR, 'resources')

PREDICTOR_FILENAME = 'shape_predictor_68_face_landmarks.dat'
PREDICTOR_FILEPATH = os.path.join(RESOURCES_DIR, PREDICTOR_FILENAME)

FACE_MODEL_FILENAME = 'face_model.txt'
FACE_MODEL_FILEPATH = os.path.join(RESOURCES_DIR, FACE_MODEL_FILENAME)

# Image processing configuration
IMAGE_UPSAMPLE_FACTOR = 1  # Ease facial landmark detection (value from dlib)
UNIVERSAL_RESIZE = 500  # [px] Photo processing size

# Lower and upper boundaries of the facial feature labels
JAW_COORDINATES = (0, 17)
LEFT_EYEBROW_COORDINATES = (17, 22)
RIGHT_EYEBROW_COORDINATES = (22, 27)
NOSE_COORDINATES = (27, 36)
RIGHT_EYE_COORDINATES = (36, 42)
LEFT_EYE_COORDINATES = (42, 48)
MOUTH_COORDINATES = (48, 68)
