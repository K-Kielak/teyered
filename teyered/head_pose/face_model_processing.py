import logging

import cv2
import numpy as np

from teyered.config import FACE_MODEL_FILEPATH, JAW_COORDINATES, \
     NOSE_COORDINATES, FACE_GROUND_TRUTH_FILEPATH

from teyered.io.files import load_image
from teyered.data_processing.points_extractor import FacialPointsExtractor


logger = logging.getLogger(__name__)


def load_face_model():
    """
    Identical to prepare face model, but returns all points instead of only the chosen ones
    :return: Face model from FACE_MODEL_FILEPATH
    """
    face_info = np.array(open(FACE_MODEL_FILEPATH, "r+").readlines(),
                         dtype=np.float32)

    face_model = np.zeros((68, 3), dtype=np.float32)
    face_model[:,0] = face_info[0:68]  # x coordinate
    face_model[:,1] = face_info[68:136]  # y coordinate
    face_model[:,2] = face_info[136:204]  # z coordinate

    logger.debug(f'Face model has been successfully loaded from {FACE_MODEL_FILEPATH}')
    return face_model

def load_ground_truth():
    """
    Get the ground truth facial points
    :return: Ground truth facial points from FACE_GROUND_TRUTH_FILEPATH
    """
    fp = FacialPointsExtractor()
    ground_truth_frame = load_image(FACE_GROUND_TRUTH_FILEPATH)

    logger.debug(f'Face model has been successfully loaded from {FACE_GROUND_TRUTH_FILEPATH}')
    return fp.detect_facial_points(ground_truth_frame)

def optimize_face_model(ground_points, generic_model_points):
    """
    Optimize the generic 3D face model in world coordinates so that it resembles the provided facial details of a person
    For example, adjust the distance between eyes, lip size etc. for the generic model points
    :param ground_points: Facial points acting as a ground truth and used to optimize the generic model
    :param generic_model_points: Generic 3D face model from load_face_model() to be optimized
    :return: np.array of optimized 3D face model and normalized ground and model points
    """
    # Shift ground points to 0 mean and range [-1,1]
    ground_points_norm = ground_points - np.mean(ground_points, axis=0) # Shift to 0 mean
    max_ground = abs(max(ground_points_norm.max(), ground_points_norm.min(), key=abs)) # max value of the array
    ground_points_norm = ground_points_norm / max_ground # Range [-1,1]

    # Shift model points to 0 mean and range [-1,1]
    generic_model_points_norm = generic_model_points - np.mean(generic_model_points, axis=0) # Shift to 0 mean
    max_model = abs(max(generic_model_points_norm.max(), generic_model_points_norm.min(), key=abs)) # max value of the array
    generic_model_points_norm = generic_model_points_norm / max_model # Range [-1,1]

    # Currently just assign the z coordinate to the facial points (doesn't work well? don't remember)
    generic_model_points_optimized = np.zeros((68, 3), dtype=np.float32)
    generic_model_points_optimized[:,0] = ground_points_norm[:,0]
    generic_model_points_optimized[:,1] = ground_points_norm[:,1]
    generic_model_points_optimized[:,2] = generic_model_points_norm[:,2]

    logger.debug('Generic model points were successfully optimized using provided ground points')
    return (generic_model_points_optimized, ground_points_norm, generic_model_points_norm)