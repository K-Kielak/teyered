import pytest
from hamcrest import *

from teyered.data_processing.blink import Blink
from teyered.data_processing.blinks_detection import detect_blinks


@pytest.mark.parametrize('measurements,expected_output', [
    (
            [(0.05, 100), (0.1, 175), (0.15, 100), (0.2, 89),
             (0.25, 76), (0.3, 50), (0.35, 25), (0.4, 44),
             (0.45, 78), (0.5, 92), (0.55, 25), (0.6, 25),
             (0.65, 100), (0.7, 125), (0.75, 100), (0.8, 100)],
            [Blink([(0.3, 50), (0.35, 25), (0.4, 44)]),
             Blink([(0.55, 25), (0.6, 25)])]
    ), (
            [(0, 100), (50, 250), (0.15, 100), (0.2, 100),
             (0.25, 100), (0.3, 100), (0.35, 70), (0.4, 0),
             (0.45, 70), (0.5, 70), (0.55, 100), (0.6, 100),
             (0.65, 100)],
            [Blink([(0.35, 70), (0.40, 0), (0.45, 70), (0.5, 70)])]
    ), (
            [(0.050, 100), (0.1, 175), (0.15, 100), (0.2, 89),
             (0.25, 76), (0.3, 50), (0.8, 25), (1.3, 44),
             (1.25, 78), (1.3, 100)],
            []
    )
])
def test_detect_blinks(measurements, expected_output):
    output = detect_blinks(measurements)
    assert_that(output, is_(equal_to(expected_output)))
