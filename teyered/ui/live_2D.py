import cv2
import numpy as np

from teyered.config import CAMERA_MATRIX, DIST_COEFFS, UNIVERSAL_RESIZE
from teyered.head_pose.camera_model import CameraModel
from teyered.io.image_processing import draw_pose_frame, draw_facial_points_frame, gray_image, resize_image, write_angles_frame, draw_projected_points, draw_projected_points_frame
from teyered.head_pose.pose import estimate_pose_live, choose_pose_points
from teyered.data_processing.points_extractor import FacialPointsExtractor
from teyered.io.files import load_video, save_video, load_image

from teyered.data_processing.eye_normalization import project_eye_points_live, compare_projected_facial, calculater_reprojection_error_live
from teyered.head_pose.face_model_processing import load_face_model, optimize_face_model, get_ground_truth

VERTICAL_LINE = "\n-----------------------\n"

TEXT_COLOR = (255,255,255)

cap = cv2.VideoCapture(0)

DISPLAY_OPTION = None

GROUND_TRUTH_FRAME = 'ground_truth/frame.jpg'


def main():
    print('\nTEYERED: live 2D')

    print(VERTICAL_LINE)
    print('1. Setting up...')
    # Setup objects
    camera_model = CameraModel()
    points_extractor = FacialPointsExtractor()

    # Setup model points
    model_points_original = load_face_model()
    facial_points_ground_truth = get_ground_truth(load_image(GROUND_TRUTH_FRAME), points_extractor)
    (model_points_optimized, _, model_points_norm) = optimize_face_model(facial_points_ground_truth, model_points_original)
    model_points = model_points_optimized # Set model points here

    pose_model_points = choose_pose_points(model_points)


    print(VERTICAL_LINE)
    print('2. Calibrating camera...')
    # Calibrate camera
    camera_model.calibrate_custom_parameters(CAMERA_MATRIX, DIST_COEFFS)

    print(VERTICAL_LINE)
    inp = input("3. Choose option (a,b,c):\n(a) Display only facial points\n(b) Display only pose\n(c) Display both\nYour choice: ")
    while(True):
        if inp == 'a' or inp == 'b' or inp == 'c':
            DISPLAY_OPTION = inp
            break
        else:
            inp = input("\nCould not parse the input. Try again: a, b or c?\nYour choice: ")

    print(VERTICAL_LINE)
    input("4. Real time view is ready. Press Enter to start...")

    previous_frame_resized = None
    previous_points = None
    prev_rvec = None
    prev_tvec = None
    frame_count = 0

    while(True):
        _, frame = cap.read()

        frame_resized = resize_image(frame)
        frame_resized_original = resize_image(frame) # Need to figure out ref vs val

        # Previous frame is None (either wasnt captured or facial points couldnt be extracted)
        if previous_frame_resized is None:
            gray_frame = gray_image(frame_resized)
            facial_points, count = points_extractor.extract_facial_points_live(None, None, gray_frame, 0)

            if facial_points is None:
                cv2.imshow('Display video', frame_resized)
                if cv2.waitKey(60) & 0xFF == ord('q'):
                    break
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
            frame_count = 1

        # Image analysis on current and previous frame
        else:
            gray_frame = gray_image(frame_resized)
            gray_previous_frame = gray_image(previous_frame_resized)
            facial_points, count = points_extractor.extract_facial_points_live(gray_previous_frame, previous_points, gray_frame, frame_count)

            # Facial points were not detected for some reason on this frame, reset everything
            if facial_points is None:
                cv2.imshow('Display video', frame_resized)
                if cv2.waitKey(60) & 0xFF == ord('q'):
                    break

                previous_frame_resized = None
                previous_points = None
                prev_rvec = None
                prev_tvec = None
                frame_count = 0
                continue

            previous_frame_resized = frame_resized_original
            previous_points = facial_points
            frame_count = count

        # Pose estimation
        r_vector, t_vector, angles, camera_world_coord = estimate_pose_live(facial_points, prev_rvec, prev_tvec, model_points)

        # Projecting eye points
        model_points_projected = project_eye_points_live (pose_model_points, r_vector, t_vector)

        # Calculate reprojection error
        error = calculater_reprojection_error_live(choose_pose_points(facial_points), model_points_projected) / UNIVERSAL_RESIZE
        print(f'Reprojection error: {error}')
        
        """
        # Calculate eye closedness
        closedness_left = compare_projected_facial(model_points_projected[36:42], facial_points[36:42])
        closedness_right = compare_projected_facial(model_points_projected[42:48], facial_points[42:48])
        print(f'Closedness left: {closedness_left}')
        print(f'Closedness right: {closedness_right}')
        """

        # Image processing
        if (DISPLAY_OPTION == 'a' or DISPLAY_OPTION == 'c'):
            draw_facial_points_frame(frame_resized, facial_points)
            draw_projected_points_frame(frame_resized, model_points_projected)
        if (DISPLAY_OPTION == 'b' or DISPLAY_OPTION == 'c'):
            draw_pose_frame(frame_resized, facial_points, r_vector, t_vector, CAMERA_MATRIX, DIST_COEFFS)
            write_angles_frame(frame_resized, angles, TEXT_COLOR)

        # Display image
        cv2.imshow('Display video', frame_resized)
        if cv2.waitKey(60) & 0xFF == ord('q'):
            break

        # Set previous vectors
        prev_rvec = r_vector
        prev_tvec = t_vector

    print(VERTICAL_LINE)
    print('5. Quitting...')
    print(VERTICAL_LINE)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
