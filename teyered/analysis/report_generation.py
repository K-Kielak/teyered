"""
This file will be deleted as this should be a part of teyered analysis
"""

import math
import sys

import cv2
import numpy as np

from teyered.config import CAMERA_MATRIX, DIST_COEFFS, UNIVERSAL_RESIZE, TRACKING_LENGTH, FRAME_TO_ANALYSE, BATCH_SIZE, PRERECORDED_VIDEO_FILEPATH
from teyered.head_pose.camera_model import CameraModel
from teyered.io.image_processing import draw_pose, draw_facial_points, display_video, gray_video, resize_video, write_angles, draw_projected_points
from teyered.head_pose.pose import estimate_pose
from teyered.data_processing.points_extractor import FacialPointsExtractor
from teyered.io.files import load_video, save_video, load_image, save_points, save_error_csv, save_facial_points_csv, save_pose_csv, initialize_csv
from teyered.data_processing.eye_normalization import project_eye_points, choose_eye_points
from teyered.data_processing.eye_normalization import project_eye_points_live, compare_projected_facial, calculater_reprojection_error
from teyered.head_pose.face_model_processing import load_face_model, optimize_face_model, get_ground_truth


VERTICAL_LINE = "\n-----------------------\n"


def main():
    print(f'{VERTICAL_LINE}\nTEYERED: Report generation')

    print(f'{VERTICAL_LINE}\n1. Setting up...')
    # Setup objects
    cap = cv2.VideoCapture(PRERECORDED_VIDEO_FILEPATH)
    camera_model = CameraModel()
    points_extractor = FacialPointsExtractor()

    # Setup model points (currently no optimization)
    model_points_original = load_face_model() # Keep original points for later use
    (model_points_optimized, _, model_points_normalized) = optimize_face_model(model_points_original, model_points_original)
    model_points = model_points_normalized # Set the actual model points here, must be normalized

    # Setup csv files in report directory
    initialize_reports()

    print(f'{VERTICAL_LINE}\n2. Calibrating camera...')
    # Calibrate camera
    camera_model.calibrate_custom_parameters(CAMERA_MATRIX, DIST_COEFFS)

    print(f'{VERTICAL_LINE}\n3. Loading and analysing video...')
    video_params = extract_video_information(cap)
    video_frames_count = video_params['frames_count']
    video_fps = video_params['fps']
    video_seconds = math.ceil(video_frames_count / video_fps)
    number_of_batches = math.ceil(video_seconds / BATCH_SIZE)

    print(f'\nVideo frames count: {video_frames_count}')
    print(f'Video fps: {video_fps}')
    print(f'Video length: {video_seconds} seconds')
    print(f'Batches: {number_of_batches}\n')
    
    if video_seconds < BATCH_SIZE:
        raise ValueError('Video must be at least BATCH_SIZE seconds long')
    
    # Last iteration's data
    last_frame = None
    last_facial_points = None
    last_r_vec = None
    last_t_vec = None
    frame_count = 0

    # Perform number_of_batches iterations each BATCH_SIZE on the video file in the path
    for x in range(0, number_of_batches):
        frames = []
        frames_count = [] # Which frames are loaded into memory
        completed = False

        # Load at most BATCH_SIZE seconds each time (todo make the loop easier)
        for i in range(x*BATCH_SIZE*video_fps, x*BATCH_SIZE*video_fps + video_fps*BATCH_SIZE):
            print_string = f'Loading frame: {int(cap.get(1))}'
            
            if cap.get(1) == video_frames_count:
                print(f'{print_string}. Video finished')
                completed = True
                break
            
            if (int(cap.get(1)) % FRAME_TO_ANALYSE) != 0:
                print(f'{print_string}. Skip')
                _, _ = cap.read()
                continue
            
            _, frame = cap.read()
            frames.append(frame)
            frames_count.append(int(cap.get(1))-1) # int(cap.get(1))-1 == i, double check
            print(f'{print_string}. Loaded')

        # Analyse loaded frames to memory
        print(f'\nANALYSING {x*BATCH_SIZE} - {(x+1)*BATCH_SIZE} seconds ({x}/{number_of_batches-1}), {len(frames)} frames\n')
        frames_resized = resize_video(frames)
        frames_gray = gray_video(frames_resized)

        # Obtain data
        facial_points_all, frame_count = points_extractor.extract_facial_points(frames_gray, previous_frame = last_frame, previous_points = last_facial_points, frame_count = frame_count)
        r_vectors_all, t_vectors_all, angles_all, _ = estimate_pose(facial_points_all, model_points, prev_rvec = last_r_vec, prev_tvec = last_t_vec)

        # Project model points from world/model coordinates to image coordinates for each frame
        model_points_projected_all = project_eye_points(frames_gray, facial_points_all, model_points, r_vectors_all, t_vectors_all)
        reprojection_errors = calculater_reprojection_error(frames_gray, facial_points_all, model_points_projected_all)

        # Calculate eye closedness
        eye_points_all = np.array([choose_eye_points(facial_points) for facial_points in facial_points_all])
        model_eye_points_projected_all = np.array([choose_eye_points(model_points_projected) for model_points_projected in model_points_projected_all])
        eye_closedness = calculate_eye_closedness(eye_points_all, model_eye_points_projected_all)
        
        (left_eye_points_all, right_eye_points_all) = [choose_eye_points(facial_points) for facial_points in facial_points_all]
        (left_model_eye_points_projected_all, right_model_eye_points_projected_all) = [choose_eye_points(model_points_projected) for model_points_projected in model_points_projected_all]
        eye_closedness_left = calculate_eye_closedness(left_eye_points_all, left_model_eye_points_projected_all)
        eye_closedness_right = calculate_eye_closedness(right_eye_points_all, right_model_eye_points_projected_all)

        # Write the intermediate data to reports on top of existing data 
        save_error_report(frames_count, reprojection_errors)
        save_pose_report(frames_count, angles_all)
        save_facial_points_report(frames_count, facial_points_all)
        save_eye_closedness_report(frames_count, eye_closedness_left, eye_closedness_right)

        # Prepare data for next batch
        last_frame = frames_gray[-1]
        last_facial_points = facial_points_all[-1]
        last_r_vec = r_vectors_all[-1]
        last_t_vec = t_vectors_all[-1]
        
        # Don't think this is necessary, as the loop would finish if condition was true anyways
        if completed:
            print('Completed')
            break

if __name__ == '__main__':
    main()
