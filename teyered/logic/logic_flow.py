# Adding relative imports (not sure if this is the correct way of doing this)
import sys
sys.path.append('../../teyered')

from data_collection.data_loading import load_image, load_video
from data_collection.data_recording import take_photo, record_video

from data_processing.facial_points_extraction import extract_facial_points
from data_processing.points_normalization import normalize_eye_points

from helpers.shape_area_calculator import calculate_polygon_area
from helpers.data_writer import write_points_to_file

def execute_teyered_logic():
    """
    Example of how the components work together
    """
    
    # Data recording or data loading
    frame = take_photo(True)

    # Data processing
    facial_points_dictionary = extract_facial_points(frame)

    if (len(facial_points_dictionary["left_eye"]) == 0 or
        len(facial_points_dictionary["right_eye"]) == 0):
        print("Could not detect eye points")
        return

    left_eye_normalized = normalize_eye_points(facial_points_dictionary["left_eye"])
    right_eye_normalized = normalize_eye_points(facial_points_dictionary["right_eye"])

    left_eye_area = calculate_polygon_area(left_eye_normalized)
    right_eye_area = calculate_polygon_area(right_eye_normalized)

    # Data interpreting, blinks, writing to file etc.
    print(left_eye_area)
    print(right_eye_area)

if __name__ == "__main__":
    execute_teyered_logic()