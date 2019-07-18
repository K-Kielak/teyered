import pytest
from hamcrest import *
import numpy as np

from teyered.data_processing.points_extractor import FacialPointsExtractor
from teyered.io.files import load_image
from teyered.io.image_processing import display_image


TEST_IMG_PATH_FACE = 'tests/test_footage/face.jpg'
TEST_IMG_PATH_FACE_2 = 'tests/test_footage/face_2.jpg'
TEST_IMG_PATH_NO_FACE = 'tests/test_footage/no_face.jpg'


# detect_facial_points()


def test_detect_facial_points_face():
    fp = FacialPointsExtractor()
    
    output = fp.detect_facial_points(load_image(TEST_IMG_PATH_FACE))

    assert_that(output, instance_of(np.ndarray))
    assert_that(output.shape, equal_to((68, 2)))

def test_detect_facial_points_no_face():
    fp = FacialPointsExtractor()
    
    output = fp.detect_facial_points(load_image(TEST_IMG_PATH_NO_FACE))

    assert_that(output, is_(None))


# track_facial_points_LK()


def test_track_facial_points_LK_type():
    fp = FacialPointsExtractor()
    frame = load_image(TEST_IMG_PATH_FACE)
    detected_points = fp.detect_facial_points(frame)

    output = fp.track_facial_points_LK(frame, frame, detected_points, detected_points)

    assert_that(output, instance_of(np.ndarray))


# extract_facial_points_live()


def test_extract_facial_points_live_type_1():
    fp = FacialPointsExtractor()
    frame = load_image(TEST_IMG_PATH_FACE)

    output = fp.extract_facial_points_live(frame)

    assert_that(output[0], instance_of(np.ndarray))
    assert_that(output[1], equal_to(1))

def test_extract_facial_points_live_type_2():
    fp = FacialPointsExtractor()
    frame = load_image(TEST_IMG_PATH_NO_FACE)

    output = fp.extract_facial_points_live(frame)

    assert_that(output[0], is_(None))
    assert_that(output[1], equal_to(0))

def test_extract_facial_points_live_type_3():
    fp = FacialPointsExtractor()
    frame = load_image(TEST_IMG_PATH_NO_FACE)

    output = fp.extract_facial_points_live(frame)

    assert_that(output[0], is_(None))
    assert_that(output[1], equal_to(0))