# Teyered

Machine Learning framework for tiredness detection

### Facial features extraction

We combine CNN facial points detection and Lukas-Kanade optical flow tracker to extract the most accurate representation of facial features at any point in time.

### Normalization of eye points

Eye area depends on various factors that are impossible to control, such as the parameters of the camera, head movement, natural eye shape and closeness to the screen to name a few. To solve this problem, we came up with an algorithm that maps eye points on a universal size grid, which can be then used to compare eyes in different environments.

### Head pose estimation

We use a single monocular RGB camera to accurately track the 3D position and rotations of the face in space. While this, in principle, is a simple PnP problem, we combine facial features extraction, Kalman filter and tuned PnP algorithm to achieve high accuracy.

### Example logic flow

```python
# Calibrate camera
camera_model = CameraModel()
camera_model.calibrate_custom_parameters(CAMERA_MATRIX, DIST_COEFFS)

# Load frames
frames = load_part_of_the_video('video.mov')

# Prepare frames for processing
frames = gray_and_resize(frames)

# Setup points extractor
points_extractor = FacialPointsExtractor()

# Extract facial points
facial_points_all = points_extractor.extract_facial_points(frames)

# Estimate pose
r_vectors_all, t_vectors_all, angles_all, camera_world_coord_all = estimate_pose(facial_points_all)

# (Restore scale/color of frames if needed)

# Draw on frames
frames = draw_facial_points(frames, facial_points_all)
frames = draw_pose(frames, facial_points_all, r_vectors_all, t_vectors_all, camera_model.get_camera_matrix(), camera_model.get_distortion_coeff())

# Show frames
display_video(frames)
```
