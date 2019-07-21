import cv2
import numpy as np

from teyered.config import CAMERA_MATRIX, DIST_COEFFS, UNIVERSAL_RESIZE
from teyered.io.image_processing import draw_pose_frame, draw_facial_points_frame, gray_image, resize_image, write_angles_frame, draw_projected_points_frame, display_image, write_closedness_frame
from teyered.io.files import load_image
from teyered.data_processing.points_extractor import FacialPointsExtractor
from teyered.data_processing.eye_normalization import project_eye_points, calculate_eye_closedness, choose_eye_points 
from teyered.head_pose.face_model_processing import load_face_model, optimize_face_model
from teyered.head_pose.camera_model import CameraModel
from teyered.head_pose.pose import estimate_pose


VERTICAL_LINE = "\n-----------------------\n"


def main():
    print(f'{VERTICAL_LINE}\nTEYERED: live 2D')

    print(f'{VERTICAL_LINE}\n1. Setting up...')
    cap = cv2.VideoCapture(0)
    camera_model = CameraModel()
    points_extractor = FacialPointsExtractor()

    # Currently no optimization for model points
    model_points_original = load_face_model()
    (model_points_optimized, _, model_points_normalized) = optimize_face_model(model_points_original, model_points_original)
    model_points = model_points_normalized # Set the actual model points here

    print(f'{VERTICAL_LINE}\n2. Calibrating camera...')
    camera_model.calibrate_custom_parameters(CAMERA_MATRIX, DIST_COEFFS)

    input(f"{VERTICAL_LINE}\n3. Real time view is ready. Press Enter to start...")

    previous_frame = None
    previous_points = None
    prev_rvec = None
    prev_tvec = None
    frame_count = 0

    LOOP = True
    while(LOOP):
        _, frame = cap.read()
        
        # Read frame and detect facial points
        frame_resized = resize_image(frame)
        gray_frame = gray_image(frame_resized)
        facial_points, frame_count = points_extractor.extract_facial_points(np.array([gray_frame]), previous_frame = previous_frame, previous_points = previous_points, frame_count = frame_count)
        facial_points = facial_points[0]

        # No facial points detected, go to the next frame
        if facial_points is None:          
            LOOP = display_image(frame_resized)

            previous_frame = None
            previous_points = None
            prev_rvec = None
            prev_tvec = None
            frame_count = 0
            continue

        # Pose estimation
        r_vector, t_vector, angles, camera_world_coord = estimate_pose(np.array([facial_points]), model_points, prev_rvec = prev_rvec, prev_tvec = prev_tvec)
        r_vector = r_vector[0]
        t_vector = t_vector[0]
        angles = angles[0]
        camera_world_coord = camera_world_coord[0]

        # Projecting eye points
        model_points_projected = project_eye_points(np.array([facial_points]), model_points, np.array([r_vector]), np.array([t_vector]))[0]

        # Calculate eye closedness
        (left_eye_points, right_eye_points) = choose_eye_points(facial_points)
        (left_model_eye_points_projected, right_model_eye_points_projected) = choose_eye_points(model_points_projected)
        eye_closedness_left = calculate_eye_closedness(np.array([left_eye_points]), np.array([left_model_eye_points_projected]))[0]
        eye_closedness_right = calculate_eye_closedness(np.array([right_eye_points]), np.array([right_model_eye_points_projected]))[0]

        # Image processing
        draw_facial_points_frame(frame_resized, facial_points)
        draw_projected_points_frame(frame_resized, model_points_projected)
        draw_pose_frame(frame_resized, facial_points, r_vector, t_vector, CAMERA_MATRIX, DIST_COEFFS)
        write_angles_frame(frame_resized, angles)
        write_closedness_frame(frame_resized, eye_closedness_left, eye_closedness_right)
        LOOP = display_image(frame_resized)

        # Set previous data
        previous_frame = gray_frame
        previous_points = facial_points
        prev_rvec = r_vector
        prev_tvec = t_vector

    print(f'{VERTICAL_LINE}\n4. Cleaning up, quitting...\n{VERTICAL_LINE}')
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
