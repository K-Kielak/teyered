import pytest
from pytest_mock import mocker
from hamcrest import *
import numpy as np

from teyered.data_processing.pose.face_model_processing import load_face_model


@pytest.fixture
def pe():
    return PoseEstimator(load_face_model())


@pytest.mark.parametrize('fp', [
    (
            np.array([])
    )
])
def test_estimate_pose_no_fp(pe, fp):
    assert_that(calling(pe.estimate_pose).with_args(fp), raises(ValueError))


@pytest.mark.parametrize('fp', [
    (
            np.full((1, 67, 2), 1)
    ), (
            np.full((1, 69, 2), 1)
    ),
])
def test_estimate_pose_wrong_shape_fp(pe, fp):
    assert_that(calling(pe.estimate_pose).with_args(fp), raises(ValueError))


@pytest.mark.parametrize('fp', [
    (
            np.full((1,), None)
    ), (
            np.full((2,), None)
    ), (
            np.full((5,), None)
    )
])
def test_estimate_pose_not_detected(mocker, pe, fp):
    # Arrange
    mocker.patch.object(pe, 'choose_pose_points')
    mocker.patch.object(pe, '_solve_pnp')
    mocker.patch.object(pe, '_get_rotation_matrix')
    mocker.patch.object(pe, '_get_euler_angles')
    mocker.patch.object(pe, '_get_camera_world_coord')

    # Act
    o_1, o_2, o_3, o_4 = pe.estimate_pose(fp)

    # Assert function output
    np.testing.assert_array_equal(o_1, fp)
    np.testing.assert_array_equal(o_2, fp)
    np.testing.assert_array_equal(o_3, fp)
    np.testing.assert_array_equal(o_4, fp)

    # Assert fp object status
    assert_that(pe._prev_rvec, is_(None))
    assert_that(pe._prev_tvec, is_(None))

    # Assert mock info
    assert_that(pe.choose_pose_points.called, equal_to(False))
    assert_that(pe._solve_pnp.called, equal_to(False))
    assert_that(pe._get_rotation_matrix.called, equal_to(False))
    assert_that(pe._get_euler_angles.called, equal_to(False))
    assert_that(pe._get_camera_world_coord.called, equal_to(False))


@pytest.mark.parametrize('fp,eo_1,eo_2,eo_3,eo_4,e_calls', [
    (
            np.random.rand(1, 2, 2), np.full((1, 3, 1), 1),
            np.full((1, 3, 1), 2), np.full((1, 3,), 3), np.full((1, 1, 3), 4),
            1
    ), (
            np.random.rand(2, 2, 2), np.full((2, 3, 1), 1),
            np.full((2, 3, 1), 2), np.full((2, 3,), 3), np.full((2, 1, 3), 4),
            2
    ), (
            np.array([np.random.rand(2, 2), None]),
            np.array([np.full((3, 1), 1), None]),
            np.array([np.full((3, 1), 2), None]),
            np.array([np.full((3,), 3), None]),
            np.array([np.full((1, 3), 4), None]), 1
    ), (
            np.array([None, np.random.rand(2, 2)]),
            np.array([None, np.full((3, 1), 1)]),
            np.array([None, np.full((3, 1), 2)]),
            np.array([None, np.full((3,), 3)]),
            np.array([None, np.full((1, 3), 4)]), 1
    ),
])
def test_estimate_pose_detected(mocker, pe, fp, eo_1, eo_2, eo_3, eo_4,
                                e_calls):
    # Arrange
    prev_rv = np.full((3, 1), 1)
    prev_tv = np.full((3, 1), 2)
    prev_cw = np.full((1, 3), 4)

    mocker.patch.object(pe, 'choose_pose_points')
    pe.choose_pose_points.return_value = None
    mocker.patch.object(pe, '_solve_pnp')
    pe._solve_pnp.return_value = prev_rv, prev_tv
    mocker.patch.object(pe, '_get_rotation_matrix')
    pe._get_rotation_matrix.return_value = (None, None)
    mocker.patch.object(pe, '_get_euler_angles')
    pe._get_euler_angles.return_value = 3, 3, 3
    mocker.patch.object(pe, '_get_camera_world_coord')
    pe._get_camera_world_coord.return_value = prev_cw

    # Act
    o_1, o_2, o_3, o_4 = pe.estimate_pose(fp)

    # Test references vs values
    prev_rv[0] = -1
    prev_tv[0] = -1
    prev_cw[0] = -1

    # Assert function output (only way to compare np.ndarray of object type
    # containing None and np.ndarray)
    for i in range(0, fp.shape[0]):
        np.testing.assert_equal(o_1[i], eo_1[i])
        np.testing.assert_equal(o_2[i], eo_2[i])
        np.testing.assert_equal(o_3[i], eo_3[i])
        np.testing.assert_equal(o_4[i], eo_4[i])

    # Assert fp object status
    if fp[-1] is None:
        assert_that(pe._prev_rvec, is_(None))
        assert_that(pe._prev_tvec, is_(None))
    else:
        # Test references vs values
        o_1[-1][0] = -1
        o_2[-1][0] = -1
        np.testing.assert_equal(pe._prev_rvec, eo_1[-1])
        np.testing.assert_equal(pe._prev_tvec, eo_2[-1])

    # Assert mock info
    assert_that(pe.choose_pose_points.call_count, equal_to(e_calls))
    assert_that(pe._solve_pnp.call_count, equal_to(e_calls))
    assert_that(pe._get_rotation_matrix.call_count, equal_to(e_calls))
    assert_that(pe._get_euler_angles.call_count, equal_to(e_calls))
    assert_that(pe._get_camera_world_coord.call_count, equal_to(e_calls))


@pytest.mark.parametrize('fp,r_vecs,t_vecs', [
    (
            np.random.rand(1, 2, 2), np.random.rand(2, 3, 2),
            np.random.rand(2, 3, 2)
    )
])
def test_pose_error_bad_input(pe, fp, r_vecs, t_vecs):
    assert_that(calling(pe.calculate_pose_reprojection_error).with_args(
            fp, r_vecs, t_vecs), raises(ValueError))