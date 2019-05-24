# Teyered

Machine Learning framework for tiredness detection.

The following sections accompany the code and explains in depth the concepts required to understand it.

## Facial features extraction

We combine CNN facial points detection and Lukas-Kanade optical flow tracker to extract the most accurate representation of facial features at any point in time.

**Why combine detection and tracking?** The noise of the facial landmarks data is partially due to the fact that the facial features are detected and not tracked. Detected - we detect facial features in each frame separately. Thus, the points marking the facial features are not necessarily "the same points" throughout the pictures. Tracking - something like Lukas-Kanade optical flow tracker would give (nearly) the same matching points, thus way less noise. Another point is that it is computationally less costly to track than to detect.

### points_extractor.py

* `_detect_facial_points(self, frame)`
    * **Description:** Detect facial points in the frame and return all the detected points in the numpy array (or None otherwise)
    * **TODO:** At the moment this is done using dlib, but we will change this to what we want

* `_track_facial_points_LK(self, previous_frame, new_frame, previous_points, detected_points)`
    * **Description:** Track the points using Lucas-Kanade optical flow. The idea is as follows: take two consecutive frames and the detected points in the first frame. Then identify the same points in the second frame and find the displacement (change) between the two frames. This uses `cv2.calcOpticalFlowPyrLK()` which calculates an optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids. Currently if any of the points gets lost, redetect all of them. Example:

    <p align="center">
        <img src="https://www.researchgate.net/publication/328750324/figure/fig3/AS:689787098910728@1541469468946/Optical-flow-map-computed-by-different-methods-a-Lucas-Kanade-b-Horn-Schunck-c.png">
    </p>

    * **TODO:** Redetect only those which are needed based on errors of individual points, for example by pure pixel distance from detected points. Try another optical flow tracking method. Try other tracking methods in general.

* `extract_facial_points(self, frames)`
    * **Description:** Simply combine detection and tracking
    * **TODO:** Introduce filtering (Kalman for real time, but maybe there are other methods that would also perform better or equally good using on prerecorded videos, but with less computational cost)

## Camera, coordinates and calibration

Different coordinate systems (different frames of reference):

* **Model (object)** - local coordinate system for a single object
* **World** - co-relating objects in 3D world. We bring objects from model to world coordinates.

<p align="center">
    <img src="https://www.ntu.edu.sg/home/ehchua/programming/opengl/images/Graphics3D_LocalSpace.png">
</p>

* **Camera** - co-relating objects in 3D world with respect to the camera
* **Image** - co-relating objects when they're projected onto a 2D plane. The relationship between  these two looks as follows:

<p align="center">
    <img src="https://www.researchgate.net/profile/Arjun_Heimsath/publication/225924677/figure/download/fig3/AS:302256164491264@1449074893202/The-paraperspective-projection-of-three-points-from-3D-world-coordinates-to-2D-image.png">
</p>

We assume pinhole camera model and use perspective transformation - converting points form world coordinates to image:

<p align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/71/Camera_Oscura.jpg/220px-Camera_Oscura.jpg">
</p>

Rotation/translation of the object from world coordinates to camera coordinates form a matrix of **extrinsic parameters**. Describes camera motion around a static scene or a rigid motion of an object in front of a still camera.

Camera has its own **intrinsic parameters** - focal length, optical center and distortion coefficient. These can be either provided by the camera manufacturer (known) or we can obtain them by calibrating the camera (`camera_model.py`). These never change as opposed to extrinsic parameters (unless we zoom in, then the focal length can change)

**TODO:** Currently, we're using approximation of the intrinsic parameters from https://www.learnopencv.com/approximate-focal-length-for-webcams-and-cell-phone-cameras/. Custom calibration using the provided class and proper calibration images should give better results than current approximation. I personally use MacBook Pro built-in camera, its parameters are not disclosed by Apple afaik. More accurate camera parameters - more accurate head pose estimation.

## Head pose estimation

We use a single monocular RGB camera to accurately track the 3D position and rotations of the face in space. While this, in principle, is a simple PnP problem, we combine facial features extraction, Kalman filter and tuned PnP algorithm to achieve high accuracy. To estimate head pose, we need:

* 2D image plane coordinates of the facial landmarks (we get this from `points_extractor.py`).
* 3D model/world (in our case model) coordinates of the face. I got this from <a href="https://github.com/TadasBaltrusaitis/OpenFace">here</a> (we will need to develop our own version as the license is incorrect for us).

This is the model both 2D and 3D coordinates use the following points in the following order (these indices are also used in the code, but they start from 1 in the below image, so of course the numbering of the array indices and the model indices will differ by 1):

<p align="center">
    <img src="data:image/png;base64,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">
</p>

### pose.py

* `_solve_pnp(image_points, model_points, prev_rvec = None, prev_tvec = None)`
    * **Description:** The function estimates the head pose: it gives the rotation and translation vectors (**extrinsic parameters**) that bring the object from its 3D model/world (in our case model) coordinates to 3D camera coordinates (and then this would get projected to the image plane and we'd get the image that we see). Maths behind PnP are explained <a href="https://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html">in the opencv docs</a>. This typically involves iterative calculations and the result isn't 100% accurate because of the nature of the equation (the algorithm is minimizing the reprojection error). We use `cv2.solvePnP()` method - to achieve better accuracy, when possible, we pass previous frame's rotation and translation vector and use what is called extrinsic guess.

    * **TODO:** Play with the parameters, see how different options influence the overall accuracy

* `_get_rotation_matrix(rotation_vector)`
    * **Description:** Rotation vector is a convenient and most compact representation of a rotation matrix (since any rotation matrix has just 3 degrees of freedom). The direction of that vector indicates the axis of rotation, while the length (or “norm”) of the vector gives the angle. A good description of this is this <a href="https://stackoverflow.com/a/13824496/7343355">stackoverflow answer</a>. Basically, it is an example of axis-angle representation (image below) and we may want to convert from this representation to rotation matrix and then euler angles (human readability or other reasons). More on this <a href="https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions">here</a>. Thus, we just use in-built function `cv2.Rodrigues()` to convert to matrix.

    <p align="center">
        <img src="https://i.stack.imgur.com/SvTtB.png">
    </p>

* `_get_euler_angles(rotation_matrix, translation_vector)`
    * **Description:** From rotation matrix to euler angles. This uses `cv2.decomposeProjectionMatrix()` which computes a decomposition of a projection matrix into a calibration and a rotation matrix and the position of a camera. It also gives Euler angles - order is not specified, I got it from <a href="https://answers.opencv.org/question/16796/computing-attituderoll-pitch-yaw-from-solvepnp/?answer=52913#post-id-52913">here</a>. Projection matrix - 3x4 matrix obtained by multiplying camera matrix by [rotation|translation] matrix, described <a href="https://answers.opencv.org/question/13545/how-to-obtain-projection-matrix/">here</a>

    <p align="center">
        <img src="https://www.researchgate.net/profile/Desmond_Fitzgerald3/publication/275771788/figure/fig1/AS:294452435406864@1447214339279/The-Euler-angles-Roll-Pitch-and-Yaw-Prior-to-magnetic-compensation-the-recorded.png">
    </p>

    * **TODO:** Double check the order of Euler angles.

* `_get_camera_world_coord(rotation_matrix, t_vector)`
    * **Description:** <a href="https://answers.opencv.org/question/133855/does-anyone-actually-use-the-return-values-of-solvepnp-to-compute-the-coordinates-of-the-camera-wrt-the-object-in-the-world-space/">Link to the explanation</a>. Basically "the return values from `cv2.solvePnP()` are the rotation and translation of the object in camera coordinate system." We can use these return values to compute camera pose w.r.t. the object in the world space: invert the rotation (transpose), then the translation is the negative of the rotated translation. For example, if the object is static and the camera is moving, we can use this to determine the camera movement.

    * **TODO:** Do 3D live and prerecorded UI for this

* `_choose_pose_points(facial_points)` and `_prepare_face_model()`
    * **Description:** Preparing face model is just extracting the relevant features from the model file (`resources/face_model.txt`). We then chose the points (obtained from the model or detection) which we will be using to solve PnP.

    * **TODO:** Put the chosen points in `config.py` for readability

* `estimate_pose(facial_points_all)`
    * **Description:** Estimate 3D pose of an object in camera coordinates from given facial points using the above methods.

    * **TODO:** Fix reference errors

**TODO:** 

* Do the head pose estimation using stereo camera setup. In this case, we'd get more accuracy. Also this would be useful in many other stuff like determining the shape of the head etc.

* Try different points for pose estimation, see which one gives the best results

* Fit the face model for each person personally using structure from motion (SfM) or other means

## Normalization of eye points

Eye area depends on various factors that are impossible to control, such as the parameters of the camera, head movement, natural eye shape and closeness to the screen to name a few. To solve this problem, we came up with an algorithm that maps eye points on a universal size grid, which can be then used to compare eyes in different environments.

Idea: we need to connect head pose with eye area estimation. This could be done by measuring the percentage difference of the "closeness" of the eyes (how much the area differs from what we percieve to be the normal 100% area). 

1. We need this "ground truth" of eye closedness - for development purposes, we will just use what we think is supposed to be ground truth, we will improve it later. 

2. We determine the head pose in a single frame: we assume that points **37, 40, 43 and 46** stay static when eyes are being closed and we don't take into account points **38, 39, 41, 42, 44, 45, 47 and 48** (which can move if we close the eyes). 

3. We simply use the obtained rotation and translation matrices to reproject points **38, 39, 41, 42, 44, 45, 47 and 48** from world/model coordinates onto the same single image and measure the percentage difference of detected points area and this reprojected "ground truth" points area. The percentage difference will indicate how closed the eyes are at any time (of course on the extremes we will lose on accuracy of this measurement, but the idea should remain).

This can be used on any frame and we don't need to concern ourselves with anything at all, only the accuracy of head pose and facial landmarks.

<p align="center">
    <img src="readme_images/eye_normalization.png">
</p>

As seen in the above photo, we can clearly notice how half closed gives a smaller area compared to ground truth. This can be done on any face angle using the rotation and translation vectors we obtained, thus accounting for all types of head movements.

* **TODO:** Fix the way coordinates are stored and optimize `face_model.txt`, as currently it's a very approximate model.

## Example logic flow

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
