import cv2
import numpy as np

from teyered.config import CAMERA_MATRIX, DIST_COEFFS, UNIVERSAL_RESIZE
from teyered.head_pose.camera_model import CameraModel
from teyered.io.image_processing import draw_pose, draw_facial_points, display_video, gray_video, resize_video, write_angles, draw_projected_points
from teyered.head_pose.pose import estimate_pose
from teyered.data_processing.points_extractor import FacialPointsExtractor
from teyered.io.files import load_video, save_video, load_image, save_points

from teyered.data_processing.eye_normalization import project_eye_points
from teyered.data_processing.eye_normalization import project_eye_points_live, compare_projected_facial, calculater_reprojection_error
from teyered.head_pose.face_model_processing import load_face_model, optimize_face_model, get_ground_truth

VIDEO_PATH = 'test_videos/video.mov'
VERTICAL_LINE = "\n-----------------------\n"

TEXT_COLOR = (255,255,255)

OUTPUT_FILE_NAME = 'output'
OUTPUT_FILE_FORMAT = 'mp4'

GROUND_TRUTH_FRAME = 'ground_truth/frame.jpg'

def main():
    print('\nTEYERED: prerecorded 2D')

    print(VERTICAL_LINE)
    print('1. Setting up...')
    # Setup objects
    camera_model = CameraModel()
    points_extractor = FacialPointsExtractor()

    # Setup model points
    model_points_original = load_face_model()
    #facial_points_ground_truth = get_ground_truth(load_image(GROUND_TRUTH_FRAME), points_extractor)
    (model_points_optimized, _, model_points_norm) = optimize_face_model(model_points_original, model_points_original)
    model_points = model_points_norm # Set model points here

    print(VERTICAL_LINE)
    print('2. Calibrating camera...')
    # Calibrate camera
    camera_model.calibrate_custom_parameters(CAMERA_MATRIX, DIST_COEFFS)

    print(VERTICAL_LINE)
    print('3. Loading and analysing video...')
    # Load and analyse video
    frames_original = load_video(VIDEO_PATH)
    frames_resized = resize_video(frames_original)
    frames = gray_video(frames_resized)
    facial_points_all, _ = points_extractor.extract_facial_points(frames)
    r_vectors_all, t_vectors_all, angles_all, camera_world_coord_all = estimate_pose(facial_points_all, model_points)
    model_points_projected_all = project_eye_points(frames, facial_points_all, model_points, r_vectors_all, t_vectors_all)

    print(VERTICAL_LINE)
    print('4. Drawing and displaying the result...')
    # Draw and display
    draw_pose(frames_resized, facial_points_all, r_vectors_all, t_vectors_all, CAMERA_MATRIX, DIST_COEFFS)
    draw_facial_points(frames_resized, facial_points_all)
    write_angles(frames_resized, angles_all, TEXT_COLOR)
    draw_projected_points(frames_resized, model_points_projected_all)
    errors = calculater_reprojection_error(frames_resized, facial_points_all, model_points_projected_all)
    save_points('errors_1.csv', errors)

    print(VERTICAL_LINE)
    input("5. Video is ready. Press Enter to view...")
    display_video(frames_resized)

    print(VERTICAL_LINE)
    print(f"6. Exporting video as {OUTPUT_FILE_NAME}.{OUTPUT_FILE_FORMAT} in working directory")
    save_video(frames_resized, OUTPUT_FILE_NAME, OUTPUT_FILE_FORMAT)

if __name__ == '__main__':
    main()
