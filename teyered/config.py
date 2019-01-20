import os


# Important paths
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCES_DIR = os.path.join(PROJECT_ROOT_DIR, 'resources')

PREDICTOR_FILENAME = 'shape_predictor_68_face_landmarks.dat'
PREDICTOR_FILEPATH = os.path.join(RESOURCES_DIR, PREDICTOR_FILENAME)
