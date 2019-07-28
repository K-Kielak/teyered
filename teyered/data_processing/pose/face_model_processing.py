import logging

import numpy as np

from teyered.config import FACE_MODEL_FILEPATH, FACE_COORDINATES_NUM


logger = logging.getLogger(__name__)


def load_face_model():
    """
    Load generic 3D face model from face_model.txt file
    :return: np.ndarray of generic 3D face model
    """
    face_info = np.array(open(FACE_MODEL_FILEPATH, "r+").readlines(),
                         dtype=np.float32)

    face_model = np.zeros((FACE_COORDINATES_NUM, 3), dtype=np.float32)
    # x, y, z coordinates
    face_model[:,0] = face_info[0:FACE_COORDINATES_NUM]
    face_model[:,1] = face_info[FACE_COORDINATES_NUM:FACE_COORDINATES_NUM*2]  
    face_model[:,2] = face_info[FACE_COORDINATES_NUM*2:FACE_COORDINATES_NUM*3]

    logger.debug(f'Face model was loaded successfully')
    return face_model