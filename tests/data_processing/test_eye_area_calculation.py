import pytest
from hamcrest import *

from teyered.config import UNIVERSAL_RESIZE
from teyered.data_processing.eye_area_calculation import calculate_polygon_area, \
    normalize_eye_points


# Universal px width for cases where x range == 11 (21-10)
PX_WIDTH = UNIVERSAL_RESIZE/11  # [px]


@pytest.mark.parametrize('test_input,expected_output', [
    (
        [(10, 10), (15, 7), (18, 5), (21, 9), (17, 12), (13, 18)],
        [(0, 500/11), (PX_WIDTH*5, 200/11), (PX_WIDTH*8, 0),
         (PX_WIDTH*11, 400/11), (PX_WIDTH*7, 700/11), (PX_WIDTH*3, 1300/11)]
    ), (
        [(15, 12), (20, 9), (23, 7), (26, 11), (22, 14), (18, 20)],
        [(0, 500/11), (PX_WIDTH*5, 200/11), (PX_WIDTH*8, 0),
         (PX_WIDTH*11, 400/11), (PX_WIDTH*7, 700/11), (PX_WIDTH*3, 1300/11)]
    ), (
        [(100, 100), (150, 70), (180, 50), (210, 90), (170, 120), (130, 180)],
        [(0, 500/11), (PX_WIDTH*5, 200/11), (PX_WIDTH*8, 0),
         (PX_WIDTH*11, 400/11), (PX_WIDTH*7, 700/11), (PX_WIDTH*3, 1300/11)]
    ), (
        [(10, 10), (15, 3), (18, 2), (21, 11), (17, 25), (13, 30)],
        [(0, 800/11), (PX_WIDTH*5, 100/11), (PX_WIDTH*8, 0),
         (PX_WIDTH*11, 900/11), (PX_WIDTH*7, 2300/11), (PX_WIDTH*3, 2800/11)]
    )
])
def test_normalize_eye_points(test_input, expected_output):
    output = normalize_eye_points(test_input)
    for (o, eo) in zip(output, expected_output):
        assert_that(o[0], is_(close_to(eo[0], 0.1)))
        assert_that(o[1], is_(close_to(eo[1], 0.1)))


@pytest.mark.parametrize('test_input,expected_output', [
    ([(10, 10), (15, 7), (18, 5), (21, 9), (17, 12), (13, 18)], 62.5),
    ([(15, 12), (20, 9), (23, 7), (26, 11), (22, 14), (18, 20)], 62.5),
    ([(100, 100), (150, 70), (180, 50), (210, 90), (170, 120), (130, 180)], 6250),
])
def test_calculate_polygon_area(test_input, expected_output):
    output = calculate_polygon_area(test_input)
    assert_that(output, is_(equal_to(expected_output)))
