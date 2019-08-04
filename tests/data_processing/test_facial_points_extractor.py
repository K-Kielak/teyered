import pytest
import numpy as np
from hamcrest import *
from pytest_mock import mocker

from teyered.data_processing.facial_points_extractor import FacialPointsExtractor


@pytest.fixture
def fp():
    return FacialPointsExtractor()


def test_extract_fp_no_frames(fp):
    assert_that(calling(fp.extract_facial_points).with_args(np.array([])),
                raises(ValueError))


@pytest.mark.parametrize('frames', [
    (
            np.random.rand(1, 2, 2)
    ), (
            np.random.rand(2, 2, 2)
    ), (
            np.random.rand(5, 2, 2)
    ), (
            np.random.rand(6, 2, 2)
    ), (
            np.random.rand(10, 2, 2)
    ), (
            np.random.rand(11, 2, 2)
    ),
])
@pytest.mark.parametrize('frame_count', [
    (
            0
    ), (
            1
    )
])
def test_extract_fp_not_detected(mocker, fp, frames, frame_count):
    # Arrange
    fp._frame_count = frame_count
    mocker.patch.object(fp, '_detect_facial_points')
    fp._detect_facial_points.return_value = None
    mocker.patch.object(fp, '_track_facial_points_LK')

    # Act
    output = fp.extract_facial_points(frames)

    # Assert function output
    np.testing.assert_array_equal(output, np.full((frames.shape[0],), None))

    # Assert fp object status
    assert_that(fp._previous_frame, is_(None))
    assert_that(fp._previous_points, is_(None))
    assert_that(fp._frame_count, equal_to(0))

    # Assert mock info
    assert_that(fp._detect_facial_points.call_count, equal_to(frames.shape[0]))
    assert_that(fp._track_facial_points_LK.called, equal_to(False))


@pytest.mark.parametrize('frames,frame_count,expected_output_last,'
                         'expected_frame_count,expected_track_called', [
    (
            np.random.rand(1, 2, 2), 0, np.full((1, 2), 1), 1, 0
    ), (
            np.random.rand(1, 2, 2), 1, np.full((1, 2), 2), 2, 1
    ), (
            np.random.rand(2, 2, 2), 0, np.full((1, 2), 2), 2, 1
    ), (
            np.random.rand(2, 2, 2), 1, np.full((1, 2), 2), 3, 2
    ), (
            np.random.rand(5, 2, 2), 0, np.full((1, 2), 2), 5, 4
    ), (
            np.random.rand(4, 2, 2), 1, np.full((1, 2), 2), 5, 4
    ), (
            np.random.rand(5, 2, 2), 1, np.full((1, 2), 1), 1, 4
    ), (
            np.random.rand(6, 2, 2), 0, np.full((1, 2), 1), 1, 4
    ), (
            np.random.rand(6, 2, 2), 1, np.full((1, 2), 2), 2, 5
    ), (
            np.random.rand(10, 2, 2), 0, np.full((1, 2), 2), 5, 8
    ), (
            np.random.rand(9, 2, 2), 1, np.full((1, 2), 2), 5, 8
    ), (
            np.random.rand(10, 2, 2), 1, np.full((1, 2), 1), 1, 8
    ), (
            np.random.rand(11, 2, 2), 0, np.full((1, 2), 1), 1, 8
    ), (
            np.random.rand(11, 2, 2), 1, np.full((1, 2), 2), 2, 9
    )
])
def test_extract_fp_detected_tracked(mocker, fp, frames, frame_count,
                                     expected_output_last,
                                     expected_frame_count,
                                     expected_track_called):
    # Arrange
    detected_points = np.full((1, 2), 1)
    tracked_points = np.full((1, 2), 2)
    fp._frame_count = frame_count
    mocker.patch.object(fp, '_detect_facial_points')
    fp._detect_facial_points.return_value = detected_points
    mocker.patch.object(fp, '_track_facial_points_LK')
    fp._track_facial_points_LK.return_value = tracked_points

    # Act
    output = fp.extract_facial_points(frames)

    # Test references vs values
    detected_points[0][0] = -1

    # Assert fp object status
    np.testing.assert_array_equal(fp._previous_frame, frames[-1])
    np.testing.assert_array_equal(fp._previous_points, expected_output_last)
    assert_that(fp._frame_count, equal_to(expected_frame_count))

    # Test references vs values
    fp._previous_points[0][0] = -1

    # Assert function output
    assert_that(output, instance_of(np.ndarray))
    assert_that(output.shape[0], equal_to(frames.shape[0]))
    assert_that(output.shape[1], equal_to(1))
    assert_that(output.shape[2], equal_to(2))
    np.testing.assert_array_equal(output[-1], expected_output_last)

    # Assert mock info
    assert_that(fp._detect_facial_points.call_count, equal_to(frames.shape[0]))
    assert_that(fp._track_facial_points_LK.call_count,
                equal_to(expected_track_called))


@pytest.mark.parametrize('frames', [
    (
            np.random.rand(1, 2, 2)
    ), (
            np.random.rand(2, 2, 2)
    ), (
            np.random.rand(5, 2, 2)
    ), (
            np.random.rand(6, 2, 2)
    ), (
            np.random.rand(10, 2, 2)
    ), (
            np.random.rand(11, 2, 2)
    ),
])
@pytest.mark.parametrize('frame_count', [
    (
            0
    ), (
            1
    )
])
def test_extract_fp_detected_not_tracked(mocker, fp, frames, frame_count):
    # Arrange
    detected_points = np.full((1, 2), 1)
    fp._frame_count = frame_count
    mocker.patch.object(fp, '_detect_facial_points')
    fp._detect_facial_points.return_value = detected_points
    mocker.patch.object(fp, '_track_facial_points_LK')
    fp._track_facial_points_LK.return_value = None

    # Act
    output = fp.extract_facial_points(frames)

    # Test references vs values
    detected_points[0][0] = -1

    # Assert fp object status
    np.testing.assert_array_equal(fp._previous_frame, frames[-1])
    np.testing.assert_array_equal(fp._previous_points, np.full((1, 2), 1))
    assert_that(fp._frame_count, equal_to(1))

    # Test references vs values
    fp._previous_points[0][0] = -1

    # Assert function output
    np.testing.assert_array_equal(output, np.full((frames.shape[0], 1, 2), 1))

    # Assert mock info
    assert_that(fp._detect_facial_points.call_count, equal_to(frames.shape[0]))
