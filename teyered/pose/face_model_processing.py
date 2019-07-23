import logging

import numpy as np

from teyered.config import FACE_MODEL_FILEPATH


logger = logging.getLogger(__name__)


def load_face_model():
    """
    Load generic 3D face model from face_model.txt file
    :return: np.ndarray of generic 3D face model
    """
    face_info = np.array(open(FACE_MODEL_FILEPATH, "r+").readlines(),
                         dtype=np.float32)

    face_model = np.zeros((68, 3), dtype=np.float32)
    face_model[:,0] = face_info[0:68]  # x coordinate
    face_model[:,1] = face_info[68:136]  # y coordinate
    face_model[:,2] = face_info[136:204]  # z coordinate

    logger.debug(f'Face model was loaded successfully')
    return face_model