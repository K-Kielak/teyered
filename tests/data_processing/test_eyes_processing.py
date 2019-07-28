import pytest
import numpy as np
from hamcrest import *
from pytest_mock import mocker

from teyered.data_processing import eyes_processing


@pytest.mark.parametrize('extr_pts_all,r_vecs_all,t_vecs_all', [
    (
            np.random.rand(1, 3, 2), np.random.rand(2, 3, 1),
            np.random.rand(1, 3, 1)
    ), (
            np.random.rand(1, 3, 2), np.random.rand(1, 3, 1),
            np.random.rand(2, 3, 1)
    ), (
            np.random.rand(1, 3, 0), np.random.rand(1, 3, 0),
            np.random.rand(1, 3, 0)
    )
])
def test_eye_closedness_bad_input(extr_pts_all, r_vecs_all, t_vecs_all):
    assert_that(calling(eyes_processing.calculate_eye_closedness).with_args(
        extr_pts_all, None, r_vecs_all, t_vecs_all), raises(ValueError))


triangle = np.array([[[1, 1], [2, 2], [3, 1]]])


@pytest.mark.parametrize('ex_pts_left,proj_pts_left,ex_pts_right,'
                         'proj_pts_right,eo_left,eo_right', [
    (
            np.full((1, 3, 2), 1), np.full((1, 3, 2), 1),
            np.full((1, 3, 2), 2), np.full((1, 3, 2), 2), np.full(1, -1),
            np.full(1, -1)
    ), (
            triangle, triangle, triangle, triangle, np.full(1, 1),
            np.full(1, 1)
    ), (
            triangle, triangle * 2, triangle, triangle * 3,
            np.full(1, (1 / 4)), np.full(1, (1 / 9))
    ), (
            np.repeat(triangle, 2, axis=0), np.repeat(triangle, 2, axis=0),
            np.repeat(triangle, 2, axis=0), np.repeat(triangle, 2, axis=0),
            np.full(2, 1), np.full(2, 1)
    ), (
            np.array([None, np.full((3, 2), 1)]),
            np.array([None, np.full((3, 2), 1)]),
            np.array([None, np.full((3, 2), 1)]),
            np.array([None, np.full((3, 2), 1)]), np.array([-1, -1]),
            np.array([-1, -1])
    )
])
def test_eye_closedness(mocker, ex_pts_left, proj_pts_left, ex_pts_right,
                        proj_pts_right, eo_left, eo_right):
    # Arrange
    mocker.patch.object(eyes_processing, '_choose_eye_points')
    eyes_processing._choose_eye_points.return_value = ex_pts_left, ex_pts_right
    mocker.patch.object(eyes_processing, '_project_eye_points')
    eyes_processing._project_eye_points.return_value = proj_pts_left, \
                                                       proj_pts_right

    # Act
    o_left, o_right = eyes_processing.calculate_eye_closedness(
        np.random.rand(1, 3, 2), np.random.rand(1, 3, 2),
        np.random.rand(1, 3, 2), np.random.rand(1, 3, 2))

    # Assert output
    np.testing.assert_array_equal(o_left, eo_left)
    np.testing.assert_array_equal(o_right, eo_right)

    # Assert mock info
    assert_that(eyes_processing._choose_eye_points.call_count, equal_to(1))
    assert_that(eyes_processing._project_eye_points.call_count, equal_to(1))


def test_polygon_area_bad_input():
    assert_that(calling(eyes_processing._calculate_polygon_area).with_args(
                        np.random.rand(1, 2, 2)), raises(ValueError))


@pytest.mark.parametrize('corner_points,expected_output', [
    (
            np.array([[1, 1], [2, 2], [3, 1]]), 1
    ), (
            np.array([[1, 1], [2, 2], [3, 1]]) * 2, 4
    ), (
            np.array([[1, 1], [1, 2], [2, 2], [2, 1]]), 1
    ), (
            np.array([[1, 1], [1, 2], [2, 2], [2, 1]]) * 2, 4
    ), (
            np.array([[0, 1], [2, 3], [4, 5]]), 0
    ), (
            np.array([[0, 1], [2, 3], [4, 5]]) * 2, 0
    )
])
def test_polygon_area(corner_points, expected_output):
    output = eyes_processing._calculate_polygon_area(corner_points)
    assert_that(output, equal_to(expected_output))
