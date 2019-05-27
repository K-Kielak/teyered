import logging

import cv2
import numpy as np

from teyered.config import FACE_MODEL_FILEPATH, JAW_COORDINATES, \
     NOSE_COORDINATES

logger = logging.getLogger(__name__)


def load_face_model():
    """
    Identical to prepare face model, but returns all points instead of only the chosen ones
    """
    face_info = np.array(open(FACE_MODEL_FILEPATH, "r+").readlines(),
                         dtype=np.float32)

    face_model = np.zeros((68, 3), dtype=np.float32)
    face_model[:,0] = face_info[0:68]  # x coordinate
    face_model[:,1] = face_info[68:136]  # y coordinate
    face_model[:,2] = face_info[136:204]  # z coordinate

    return face_model

def optimize_face_model(facial_points, model_points):
    """
    Optimize 3D face model so that it resembles the photo face
    :param facial_points: Facial points ground truth
    :param model_points: Generic 3D face model to be optimized
    :return: 3D face model adjusted by the facial points
    """
    
    # Shift facial points to 0 mean and range [-1,1]
    mean_facial = np.mean(facial_points, axis=0)
    facial_points_norm = facial_points - mean_facial
    max_facial = abs(max(facial_points_norm.max(), facial_points_norm.min(), key=abs)) # max value of the array
    facial_points_norm = facial_points_norm / max_facial

    # Shift model points to 0 mean and range [-1,1]
    mean_model = np.mean(model_points, axis=0)
    model_points_norm = model_points - mean_model
    max_model = abs(max(model_points_norm.max(), model_points_norm.min(), key=abs))
    model_points_norm = model_points_norm / max_model

    # Currently just assign the z coordinate to the facial points (doesn't work)
    model_points_optimized = np.zeros((68, 3), dtype=np.float32)
    model_points_optimized[:,0] = facial_points_norm[:,0]
    model_points_optimized[:,1] = facial_points_norm[:,1]
    model_points_optimized[:,2] = model_points_norm[:,2]

    return (model_points_optimized, facial_points_norm, model_points_norm)

def get_ground_truth(frame, facial_ponts_extractor):
    """
    Get the ground truth facial points
    :param frame: Ground truth frame
    :return: Ground truth facial points
    """

    return facial_ponts_extractor.detect_facial_points(frame)