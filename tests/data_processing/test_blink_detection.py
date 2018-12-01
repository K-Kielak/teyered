from datetime import datetime, timedelta

import pytest
from hamcrest import *

from teyered.data_processing.blink import Blink
from teyered.data_processing.blinks_detection import detect_blinks


def datetime_from_ms(ms):
    return datetime(1970, 1, 1) + timedelta(milliseconds=ms)


@pytest.mark.parametrize('measurements,expected_output', [
    (
            [(datetime_from_ms(50), 100), (datetime_from_ms(100), 175),
             (datetime_from_ms(150), 100), (datetime_from_ms(200), 89),
             (datetime_from_ms(250), 76), (datetime_from_ms(300), 50),
             (datetime_from_ms(350), 25), (datetime_from_ms(400), 44),
             (datetime_from_ms(450), 78), (datetime_from_ms(500), 92),
             (datetime_from_ms(550), 25), (datetime_from_ms(600), 25),
             (datetime_from_ms(650), 100), (datetime_from_ms(700), 125),
             (datetime_from_ms(750), 100), (datetime_from_ms(800), 100)],
            [Blink([(datetime_from_ms(300), 50),
                    (datetime_from_ms(350), 25),
                    (datetime_from_ms(400), 44)]),
             Blink([(datetime_from_ms(550), 25),
                    (datetime_from_ms(600), 25)])]
    ), (
            [(datetime_from_ms(0), 100), (datetime_from_ms(50), 250),
             (datetime_from_ms(150), 100), (datetime_from_ms(200), 100),
             (datetime_from_ms(250), 100), (datetime_from_ms(300), 100),
             (datetime_from_ms(350), 70), (datetime_from_ms(400), 0),
             (datetime_from_ms(450), 70), (datetime_from_ms(500), 70),
             (datetime_from_ms(550), 100), (datetime_from_ms(600), 100),
             (datetime_from_ms(650), 100)],
            [Blink([(datetime_from_ms(350), 70),
                    (datetime_from_ms(400), 0),
                    (datetime_from_ms(450), 70),
                    (datetime_from_ms(500), 70)])]
    ), (
            [(datetime_from_ms(50), 100), (datetime_from_ms(100), 175),
             (datetime_from_ms(150), 100), (datetime_from_ms(200), 89),
             (datetime_from_ms(250), 76), (datetime_from_ms(300), 50),
             (datetime_from_ms(800), 25), (datetime_from_ms(1300), 44),
             (datetime_from_ms(1250), 78), (datetime_from_ms(1300), 100)],
            []
    )
])
def test_detect_blinks(measurements, expected_output):
    output = detect_blinks(measurements)
    assert_that(output, is_(equal_to(expected_output)))
