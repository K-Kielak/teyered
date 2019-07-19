import cv2
import numpy as np

from teyered.config import CAMERA_MATRIX, DIST_COEFFS, PRERECORDED_VIDEO_FILEPATH
from teyered.data_processing.points_extractor import FacialPointsExtractor
from teyered.data_processing.eye_normalization import project_eye_points
from teyered.head_pose.pose import estimate_pose
from teyered.head_pose.face_model_processing import load_face_model, optimize_face_model
from teyered.head_pose.camera_model import CameraModel
from teyered.io.image_processing import draw_pose, draw_facial_points, display_video, gray_video, resize_video, write_angles, draw_projected_points
from teyered.io.files import load_video, save_video


VERTICAL_LINE = "\n-----------------------\n"

OUTPUT_FILE_NAME = 'output'
OUTPUT_FILE_FORMAT = 'mp4'


def main():
    print(f'{VERTICAL_LINE}\nTEYERED: live 2D')

    print(f'{VERTICAL_LINE}\n1. Setting up...')
    # Setup objects
    camera_model = CameraModel()
    points_extractor = FacialPointsExtractor()

    # Setup model points (currently no optimization)
    model_points_original = load_face_model() # Keep original points for later use
    (model_points_optimized, _, model_points_normalized) = optimize_face_model(model_points_original, model_points_original)
    model_points = model_points_normalized # Set the actual model points here, must be normalized

    print(f'{VERTICAL_LINE}\n2. Calibrating camera...')
    # Calibrate camera
    camera_model.calibrate_custom_parameters(CAMERA_MATRIX, DIST_COEFFS)

    print(f'{VERTICAL_LINE}\n3. Loading and analysing video...')
    # Load and analyse video
    frames_original = load_video(PRERECORDED_VIDEO_FILEPATH)
    frames_resized = resize_video(frames_original)
    frames = gray_video(frames_resized)
    facial_points_all, _ = points_extractor.extract_facial_points(frames)
    r_vectors_all, t_vectors_all, angles_all, camera_world_coord_all = estimate_pose(facial_points_all, model_points)
    model_points_projected_all = project_eye_points(frames, facial_points_all, model_points, r_vectors_all, t_vectors_all)

    print(f'{VERTICAL_LINE}\n4. Drawing and displaying the result...')
    # Draw and display
    draw_pose(frames_resized, facial_points_all, r_vectors_all, t_vectors_all)
    draw_facial_points(frames_resized, facial_points_all)
    write_angles(frames_resized, angles_all, TEXT_COLOR)
    draw_projected_points(frames_resized, model_points_projected_all)

    input(f'{VERTICAL_LINE}\n5. Video is ready. Press Enter to view...')
    display_video(frames_resized)

    print(f'{VERTICAL_LINE}\n6. Exporting video as {OUTPUT_FILE_NAME}.{OUTPUT_FILE_FORMAT} in working directory')
    save_video(frames_resized, OUTPUT_FILE_NAME, OUTPUT_FILE_FORMAT)

    print(f'{VERTICAL_LINE}\n7. Quitting...\n{VERTICAL_LINE}')
    # Clean up the resources here

if __name__ == '__main__':
    main()
