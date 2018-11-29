from datetime import timedelta
from math import sqrt

from teyered.data_processing.blink import Blink

# How many standard deviations below eyes average should indicate
# high/low threshold for a blink in double thresholding
HIGH_THRESHOLD = 1.23
LOW_THRESHOLD = 0.5

# All blinks with duration higher than the constant below will be filtered
MAX_BLINK_DURATION = 1000  # [ms]


def detect_blinks(eye_measurements):
    """
    Detects blinks from eye heights measurements by finding continuous
    sets of outlier heights below the eyes average using double thresholding
    :param eye_measurements: List of tuples such as
    (time_of_the_measurement, height_of_the_eye)
    :return: List of blinks that don't exceed 1s
    """
    _, heights = zip(*eye_measurements)
    are_blinks = _find_blink_heights(heights)
    blinks = _convert_blink_heights_to_blinks(are_blinks, eye_measurements)
    blinks = [b for b in blinks
              if b.get_duration() < timedelta(milliseconds=MAX_BLINK_DURATION)]

    return blinks


def _find_blink_heights(heights):
    average_height = sum(heights) / len(heights)
    variance = sum((height - average_height) ** 2
                   for height in heights) / len(heights)
    standard_deviation = sqrt(variance)

    high_threshold = average_height - HIGH_THRESHOLD * standard_deviation
    low_threshold = average_height - LOW_THRESHOLD * standard_deviation

    are_blinks = [h < high_threshold for h in heights]
    for i in range(1, len(heights)):
        are_blinks[i] = are_blinks[i] or \
                        (heights[i] < low_threshold and are_blinks[i - 1])

    for i in range(len(heights) - 1, 0, -1):
        are_blinks[i] = are_blinks[i] or \
                        (heights[i] < low_threshold and are_blinks[i + 1])

    return are_blinks


def _convert_blink_heights_to_blinks(are_blinks, measurements):
    blinks = []
    blink_measurements = []
    for i, is_blink in enumerate(are_blinks):
        if is_blink:
            blink_measurements.append(measurements[i])
        elif blink_measurements:
            blinks.append(Blink(blink_measurements))
            blink_measurements = []

    return blinks
