import pytest
from hamcrest import *
import numpy as np

from teyered.io.files import load_video, load_image

TEST_IMG_PATH = 'tests/test_images/face.jpg'
TEST_VIDEO_PATH = 'tests/test_images/video.mp4'

def test_load_video_type():
    output = load_video(TEST_VIDEO_PATH)

    assert_that(output, instance_of(np.ndarray))

def test_load_image_type():
    output = load_image(TEST_IMG_PATH)

    assert_that(output, instance_of(np.ndarray))