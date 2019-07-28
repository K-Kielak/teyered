import cv2
import pytest
import numpy as np
from hamcrest import *
from pytest_mock import mocker

from teyered.data_processing.projection import project_points
from teyered.data_processing.projection import calculate_reprojection_error


@pytest.mark.parametrize('r_vecs_all,t_vecs_all', [
    (
            np.random.rand(1, 3, 1), np.random.rand(1, 4, 1)
    ), (
            np.random.rand(1, 3, 0), np.random.rand(1, 3, 0)
    ),
])
def test_project_points_bad_input(r_vecs_all, t_vecs_all):
    assert_that(calling(project_points).with_args(
        None, r_vecs_all, t_vecs_all), raises(ValueError))


@pytest.mark.parametrize('proj_points,r_vecs_all,t_vecs_all,expected_output', [
    (
            np.random.rand(3, 2), np.random.rand(1, 3, 1),
            np.random.rand(1, 3, 1), np.full((1, 3, 2), 1)
    ), (
            np.random.rand(3, 2), np.random.rand(2, 3, 1),
            np.random.rand(2, 3, 1), np.full((2, 3, 2), 1)
    ), (
            np.random.rand(3, 2), np.array([None]), np.array([None]),
            np.array([None])
    ), (
            np.random.rand(3, 2), np.array([None, np.full((3, 1), 1)]),
            np.array([None, np.full((3, 1), 1)]),
            np.array([None, np.full((3, 2), 1)])
    ), (
            np.random.rand(3, 2), np.array([None, np.full((3, 1), 1)]),
            np.array([np.full((3, 1), 1), None]), np.array([None, None])
    )
])
def test_project_points(mocker, proj_points, r_vecs_all, t_vecs_all,
                        expected_output):
    mocker.patch('cv2.projectPoints')
    cv2.projectPoints.return_value = np.full((3, 1, 2), 1), None
    output = project_points(proj_points, r_vecs_all, t_vecs_all)
    # Only way to compare np.ndarray of object type containing
    # None and np.ndarray
    for i in range(0, r_vecs_all.shape[0]):
        np.testing.assert_array_equal(output[i], expected_output[i])


@pytest.mark.parametrize('gr_pts_all,pr_pts_all', [
    (
            np.random.rand(1, 3, 2), np.random.rand(1, 3, 3)
    ), (
            np.random.rand(1, 3, 0), np.random.rand(1, 3, 0)
    ),
])
def test_reprojection_error_bad_input(gr_pts_all, pr_pts_all):
    assert_that(calling(calculate_reprojection_error).with_args(
            gr_pts_all, pr_pts_all), raises(ValueError))


@pytest.mark.parametrize('gr_pts_all,pr_pts_all,expected_error', [
    (
            np.array([[[1, 2]]]), np.array([[[4, 6]]]), np.array([5])
    ), (
            np.array([[[1, 2], [4, 6]]]), np.array([[[4, 6], [1, 2]]]),
            np.array([10])
    ), (
            np.array([[[1, 2]], [[4, 6]]]), np.array([[[4, 6]], [[1, 2]]]),
            np.array([5, 5])
    ), (
            np.array([None]), np.array([None]), np.array([-1])
    ), (
            np.array([None, np.array([[4, 6]])]),
            np.array([None, np.array([[1, 2]])]), np.array([-1, 5])
    ), (
            np.array([None, np.array([[4, 6]])]),
            np.array([np.array([[1, 2]]), None]), np.array([-1, -1])
    )
])
def test_reprojection_error(gr_pts_all, pr_pts_all, expected_error):
    output = calculate_reprojection_error(gr_pts_all, pr_pts_all)
    np.testing.assert_array_equal(output, expected_error)
