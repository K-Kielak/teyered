import cv2
import numpy as np

from teyered.config import CAMERA_MATRIX, DIST_COEFFS, UNIVERSAL_RESIZE
from teyered.io.image_processing import draw_pose_frame, draw_facial_points_frame, gray_image, resize_image, write_angles_frame, draw_projected_points_frame, display_image
from teyered.io.files import load_image
from teyered.data_processing.points_extractor import FacialPointsExtractor
from teyered.data_processing.eye_normalization import project_eye_points_live, calculate_eye_closedness_live, choose_eye_points 
from teyered.head_pose.face_model_processing import load_face_model, optimize_face_model
from teyered.head_pose.camera_model import CameraModel
from teyered.head_pose.pose import estimate_pose_live


VERTICAL_LINE = "\n-----------------------\n"


def main():
    print(f'{VERTICAL_LINE}\nTEYERED: live 2D')

    print(f'{VERTICAL_LINE}\n1. Setting up...')
    # Setup objects
    cap = cv2.VideoCapture(0)
    camera_model = CameraModel()
    points_extractor = FacialPointsExtractor()

    # Setup model points (currently no optimization)
    model_points_original = load_face_model() # Keep original points for later use
    (model_points_optimized, _, model_points_normalized) = optimize_face_model(model_points_original, model_points_original)
    model_points = model_points_normalized # Set the actual model points here, must be normalized

    print(f'{VERTICAL_LINE}\n2. Calibrating camera...')
    # Calibrate camera
    camera_model.calibrate_custom_parameters(CAMERA_MATRIX, DIST_COEFFS)

    inp = input(f"{VERTICAL_LINE}\n3. Choose option (a,b,c):\n(a) Display only facial points\n(b) Display only pose\n(c) Display both\nYour choice: ")
    # Iterate until the correct option is chosen
    DISPLAY_OPTION = None
    while(DISPLAY_OPTION is None):
        if inp == 'a' or inp == 'b' or inp == 'c':
            DISPLAY_OPTION = inp
        else:
            inp = input("\nCould not parse the input. Try again: a, b or c?\nYour choice: ")

    input(f"{VERTICAL_LINE}\n4. Real time view is ready. Press Enter to start...")

    previous_frame_resized = None
    previous_points = None
    prev_rvec = None
    prev_tvec = None
    frame_count = 0

    LOOP = True
    while(LOOP):
        _, frame = cap.read()

        frame_resized = resize_image(frame)
        frame_resized_original = frame_resized # No time to test, better safe than sorry

        # Previous frame is None (either wasnt captured or facial points couldnt be extracted)
        if previous_frame_resized is None or previous_points is None:
            gray_frame = gray_image(frame_resized)
            facial_points, frame_count = points_extractor.extract_facial_points_live(gray_frame)
            
            if facial_points is None:                
                LOOP = display_image(frame_resized)

                previous_frame_resized = None
                previous_points = None
                prev_rvec = None
                prev_tvec = None
                frame_count = 0
                continue

            previous_frame_resized = frame_resized_original
            previous_points = facial_points
            prev_rvec = None
            prev_tvec = None

        # Image analysis on current and previous frame
        else:
            gray_frame = gray_image(frame_resized)
            gray_previous_frame = gray_image(previous_frame_resized)
            facial_points, frame_count = points_extractor.extract_facial_points_live(gray_frame, gray_previous_frame, previous_points, frame_count)

            # Facial points were not detected for some reason on this frame, reset everything
            if facial_points is None:
                LOOP = display_image(frame_resized)

                previous_frame_resized = None
                previous_points = None
                prev_rvec = None
                prev_tvec = None
                frame_count = 0
                continue

            previous_frame_resized = frame_resized_original
            previous_points = facial_points

        # Pose estimation
        r_vector, t_vector, angles, camera_world_coord = estimate_pose_live(facial_points, model_points, prev_rvec, prev_tvec)

        # Projecting eye points
        model_points_projected = project_eye_points_live(model_points, r_vector, t_vector)

        # Calculate eye closedness
        (left_eye_points, right_eye_points) = choose_eye_points(facial_points)
        (left_model_eye_points_projected, right_model_eye_points_projected) = choose_eye_points(model_points_projected)
        eye_closedness_left = calculate_eye_closedness_live(left_eye_points, left_model_eye_points_projected)
        eye_closedness_right = calculate_eye_closedness_live(right_eye_points, right_model_eye_points_projected)
        print(f'Eye closedness: {eye_closedness_right}%, {eye_closedness_left}%')

        # Image processing
        if (DISPLAY_OPTION == 'a' or DISPLAY_OPTION == 'c'):
            draw_facial_points_frame(frame_resized, facial_points)
            draw_projected_points_frame(frame_resized, model_points_projected)
        if (DISPLAY_OPTION == 'b' or DISPLAY_OPTION == 'c'):
            draw_pose_frame(frame_resized, facial_points, r_vector, t_vector, CAMERA_MATRIX, DIST_COEFFS)
            write_angles_frame(frame_resized, angles)

        # Display image
        LOOP = display_image(frame_resized)

        # Set previous vectors
        prev_rvec = r_vector
        prev_tvec = t_vector

    print(f'{VERTICAL_LINE}\n5. Quitting...\n{VERTICAL_LINE}')
    # Clean up the resources here
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
