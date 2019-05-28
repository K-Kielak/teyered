import math
import sys

import cv2
import numpy as np

from teyered.config import CAMERA_MATRIX, DIST_COEFFS, UNIVERSAL_RESIZE, TRACKING_LENGTH
from teyered.head_pose.camera_model import CameraModel
from teyered.io.image_processing import draw_pose, draw_facial_points, display_video, gray_video, resize_video, write_angles, draw_projected_points
from teyered.head_pose.pose import estimate_pose
from teyered.data_processing.points_extractor import FacialPointsExtractor
from teyered.io.files import load_video, save_video, load_image, save_points, save_error_csv, save_facial_points_csv, save_pose_csv, initialize_csv
from teyered.data_processing.eye_normalization import project_eye_points
from teyered.data_processing.eye_normalization import project_eye_points_live, compare_projected_facial, calculater_reprojection_error
from teyered.head_pose.face_model_processing import load_face_model, optimize_face_model, get_ground_truth


VIDEO_PATH = 'test_videos/video.mov'
VERTICAL_LINE = "\n-----------------------\n"

TEXT_COLOR = (255,255,255)

EYE_DATA_PATH = 'eye_data.csv'
POSE_DATA_PATH = 'pose_data.csv'
ERROR_DATA_PATH = 'errors.csv'

BATCH_SIZE = 10 # seconds, each iteration analyse BATCH_SIZE seconds worth of footage
FRAME_TO_ANALYSE = 1 # Analyse every FRAME_TO_ANALYSEth frame. = 1, then analyse every frame. = 2, then analyse every second frame etc.


def main():
    print('\nTEYERED: prerecorded 2D')

    print(VERTICAL_LINE)
    print('1. Setting up...')

    # Setup objects
    camera_model = CameraModel()
    points_extractor = FacialPointsExtractor()

    # Setup model points
    model_points_original = load_face_model()
    (model_points_optimized, _, model_points_norm) = optimize_face_model(model_points_original, model_points_original)
    model_points = model_points_norm

    # Setup csv files
    initialize_csv(ERROR_DATA_PATH, POSE_DATA_PATH, EYE_DATA_PATH)

    print(VERTICAL_LINE)
    print('2. Calibrating camera...')

    # Calibrate camera
    camera_model.calibrate_custom_parameters(CAMERA_MATRIX, DIST_COEFFS)

    print(VERTICAL_LINE)
    print('3. Loading and analysing video...')

    # Get video information
    cap = cv2.VideoCapture(VIDEO_PATH)
    video_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_seconds = math.ceil(video_frames_count/video_fps)
    number_of_batches = math.ceil(video_seconds / BATCH_SIZE)

    print(f'\nVideo frames count: {video_frames_count}')
    print(f'Video fps: {video_fps}')
    print(f'Video length: {video_seconds} seconds')
    print(f'Batches: {number_of_batches}\n')
    
    if video_seconds < BATCH_SIZE:
        print('Video must be at least BATCH_SIZE seconds long')
        sys.exit(0)
    
    # Analyse video
    last_frame = None
    last_facial_points = []
    last_r_vec = []
    last_t_vec = []
    frame_count = 0

    for x in range(0, number_of_batches):
        frames = []
        frames_count = []
        completed = False

        # Load at most BATCH_SIZE seconds each time
        for i in range(x*BATCH_SIZE*video_fps, x*BATCH_SIZE*video_fps + video_fps*BATCH_SIZE):
            print_string = f'Loading frame: {int(cap.get(1))}'
            
            if cap.get(1) == video_frames_count:
                print(f'{print_string}. Video finished')
                completed = True
                break
            
            if (int(cap.get(1)) % FRAME_TO_ANALYSE) != 0:
                print(f'{print_string}. Skip')
                ret, frame = cap.read()
                continue
            
            ret, frame = cap.read()
            frames.append(frame)
            frames_count.append(int(cap.get(1))-1)
            print(f'{print_string}. Loaded')

        # Analyse loaded frames
        print(f'\nANALYSING {x*BATCH_SIZE} - {(x+1)*BATCH_SIZE} seconds ({x}/{number_of_batches-1}), {len(frames)} frames\n')
        
        frames_resized = resize_video(frames)
        frames = gray_video(frames_resized)

        if last_frame is None or last_facial_points is []:
            facial_points_all, frame_count = points_extractor.extract_facial_points(frames, frame_count=frame_count)
            r_vectors_all, t_vectors_all, angles_all, _ = estimate_pose(facial_points_all, model_points)

            model_points_projected_all = project_eye_points(frames, facial_points_all, model_points, r_vectors_all, t_vectors_all)
            errors = calculater_reprojection_error(frames, facial_points_all, model_points_projected_all)

            save_facial_points_csv(EYE_DATA_PATH, frames_count, facial_points_all)
            save_pose_csv(POSE_DATA_PATH, frames_count, angles_all)
            save_error_csv(ERROR_DATA_PATH, frames_count, errors)

            last_frame = frames[-1]
            last_facial_points = facial_points_all[-1]
            last_r_vec = r_vectors_all[-1]
            last_t_vec = t_vectors_all[-1]

        else:
            facial_points_all, frame_count = points_extractor.extract_facial_points(frames, previous_frame=last_frame, previous_points=last_facial_points, frame_count=frame_count)
            r_vectors_all, t_vectors_all, angles_all, _ = estimate_pose(facial_points_all, model_points, prev_rvec=last_r_vec, prev_tvec=last_t_vec)

            model_points_projected_all = project_eye_points(frames, facial_points_all, model_points, r_vectors_all, t_vectors_all)
            errors = calculater_reprojection_error(frames, facial_points_all, model_points_projected_all)

            save_facial_points_csv(EYE_DATA_PATH, frames_count, facial_points_all)
            save_pose_csv(POSE_DATA_PATH, frames_count, angles_all)
            save_error_csv(ERROR_DATA_PATH, frames_count, errors)

            last_frame = frames[-1]
            last_facial_points = facial_points_all[-1]
            last_r_vec = r_vectors_all[-1]
            last_t_vec = t_vectors_all[-1]

        if completed:
            print('Completed')
            break

if __name__ == '__main__':
    main()
