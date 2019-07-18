import pytest
from hamcrest import *
import numpy as np

from teyered.head_pose.pose import 
from teyered.io.files import load_image


TEST_IMG_PATH_FACE = 'tests/test_footage/face.jpg'
TEST_IMG_PATH_FACE_2 = 'tests/test_footage/face_2.jpg'
TEST_IMG_PATH_NO_FACE = 'tests/test_footage/no_face.jpg'