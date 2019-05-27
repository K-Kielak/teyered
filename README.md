# Teyered

Machine Learning framework for tiredness detection.

The following sections accompany the code and explain in depth the concepts required to understand it.

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

## Face model and coordinates

Dlib uses the following convention for face points:

<p align="center">
    <img src="https://www.pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup.jpg" width=500>
</p>

* **Jaw:** [1-17]
* **Left eyebrow:** [18-22]
* **Right eyebrow:** [23-27]
* **Nose:** [28-36] (**tip of the nose:** 31)
* **Left eye:** [37-42]
* **Right eye:** [43-48]
* **Mouth:** [49-64]

Image plane coordinates are as follows (px, 500 stands for `UNIVERSAL_RESIZE` variable in `config.py`):

<p align="center">
    <img src="readme_images/image_plane_coord.png" width=200>
</p>


We use 3D face model from <a href="https://github.com/TadasBaltrusaitis/OpenFace">here</a> (we will need to develop our own version as the license is incorrect for us).

<p align="center">
    <img src="readme_images/face_3D.gif" width=400>
</p>

The following are the `face_model.txt` coordinates:

<p align="center">
    <img src="readme_images/face_3D_coord.png">
</p>

Thus, to plot the head pose orientation in a "human friendly way", we plot lines from (0,0,0) to (1,0,0), (0,-1,0) and (0,0,-1) in both 2D and 3D. No transformation would mean having 0&deg; pitch, roll and yaw.

The following are the `face_model.txt` and 2D facial points on image plane coordinates to illustrate the relation:

<p align="center">
    <img src="readme_images/face_3D_2D.png">
</p>

* Red: x-axis
* Green: y-axis
* Blue - z-axis

We could rotate both facial points and model points 180&deg; around x-axis (red) to get a more "human readable and intuitive" representation of both angles and axis, but this is not necessary. Model is adjusted according to the way image plane works (and thus facial points are obtained).

### face_model_processing.py

* `optimize_face_model(facial_points, model_points)`
    * **Description:** Optimize the model points so that the shape resembles the one of the person (for example: eye width, eye-nose-eye distance etc.). Currently, it only normalizes the coordinates of the points (shift and scale) and adds the third dimension to facial points.
    
    * **TODO:** Delete and rewrite to something more sophisticated

* `get_ground_truth(frame, facial_ponts_extractor)`
    * **Description:** Dummy function to detect the facial points from a frame where roll, yaw and pitch are all supposed to be 0 ("ground truth" picture).

    * **TODO:** Delete and rewrite to something more sophisticated

**TODO:** Move some methods from pose.py (for example loading the head pose from the `.txt` file)

## Head pose estimation

We use a single monocular RGB camera to accurately track the 3D position and rotations of the face in space. While this, in principle, is a simple PnP problem, we combine facial features extraction, Kalman filter and tuned PnP algorithm to achieve high accuracy. To estimate head pose, we need:

* 2D image plane coordinates of the facial landmarks (we get this from `points_extractor.py`).
* 3D model/world (in our case model) coordinates of the face.

Points we currently use for head pose estimation (these are the points which we assume only change when head pose changes):

* **Jaw:** [1-17]
* **Nose:** [28-36]
* **Left eye (corners):** 37, 40
* **Right eye (corners):** 43, 46
* **Mouth (corners):** 49, 55

Thus, other points are assumed to have no effect on head pose and could be changed at static head pose (for example - moving eyebrows up and down while not moving the head).

### pose.py

* `_solve_pnp(image_points, model_points, prev_rvec = None, prev_tvec = None)`
    * **Description:** The function estimates the head pose: it gives the rotation and translation vectors (**extrinsic parameters**) that bring the object from its 3D model/world (in our case model) coordinates to 3D camera coordinates (and then this would get projected to the image plane and we'd get the image that we see). Maths behind PnP are explained <a href="https://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html">in the opencv docs</a>. This typically involves iterative calculations and the result isn't 100% accurate because of the nature of the equation (the algorithm is minimizing the reprojection error). We use `cv2.solvePnP()` method - to achieve better accuracy, when possible, we pass previous frame's rotation and translation vector and use what is called extrinsic guess.

    * **TODO:** Play with the parameters, see how different options influence the overall accuracy

* `_get_rotation_matrix(rotation_vector)`
    * **Description:** Rotation vector is a convenient and most compact representation of a rotation matrix (since any rotation matrix has just 3 degrees of freedom). The direction of that vector indicates the axis of rotation, while the length (or “norm”) of the vector gives the angle. A good description of this is this <a href="https://stackoverflow.com/a/13824496/7343355">stackoverflow answer</a>. Basically, it is an example of axis-angle representation (image below) and we may want to convert from this representation to rotation matrix and then euler angles (human readability or other reasons). More on this <a href="https://en.wikipedia.org/wiki/Rotation_formalisms_in_three_dimensions">here</a>. Thus, we just use in-built function `cv2.Rodrigues()` to convert to matrix.

    <p align="center">
        <img src="https://i.stack.imgur.com/SvTtB.png" width=250>
    </p>

* `_get_euler_angles(rotation_matrix, translation_vector)`
    * **Description:** From rotation matrix to euler angles. This uses `cv2.decomposeProjectionMatrix()` which computes a decomposition of a projection matrix into a calibration and a rotation matrix and the position of a camera. It also gives Euler angles - order is not specified, I got it from <a href="https://answers.opencv.org/question/16796/computing-attituderoll-pitch-yaw-from-solvepnp/?answer=52913#post-id-52913">here</a>. Projection matrix - 3x4 matrix obtained by multiplying camera matrix by [rotation|translation] matrix, described <a href="https://answers.opencv.org/question/13545/how-to-obtain-projection-matrix/">here</a>

    <p align="center">
        <img src="https://www.researchgate.net/profile/Desmond_Fitzgerald3/publication/275771788/figure/fig1/AS:294452435406864@1447214339279/The-Euler-angles-Roll-Pitch-and-Yaw-Prior-to-magnetic-compensation-the-recorded.png">
    </p>

    * **TODO:** Check the order of Euler angles (pretty sure yaw, pitch, roll are ZYX)

* `_get_camera_world_coord(rotation_matrix, t_vector)`
    * **Description:** <a href="https://answers.opencv.org/question/133855/does-anyone-actually-use-the-return-values-of-solvepnp-to-compute-the-coordinates-of-the-camera-wrt-the-object-in-the-world-space/">Link to the explanation</a>. Basically "the return values from `cv2.solvePnP()` are the rotation and translation of the object in camera coordinate system." We can use these return values to compute camera pose w.r.t. the object in the world space: invert the rotation (transpose), then the translation is the negative of the rotated translation. For example, if the object is static and the camera is moving, we can use this to determine the camera movement.

    * **TODO:** Do 3D live and prerecorded UI for this

* `_choose_pose_points(facial_points)` and `_prepare_face_model()`
    * **Description:** Extracting the relevant points for pose estimation from the model file (`resources/face_model.txt`) which was described earlier in this section. We then chose the points (obtained from the model or detection) which we will be using to solve PnP.

    * **TODO:** Put the chosen points in `config.py` for readability

* `estimate_pose(facial_points_all)`
    * **Description:** Estimate 3D pose of an object in camera coordinates from given facial points using the above methods.

    * **TODO:** Fix reference errors

**TODO:** 

* Do the head pose estimation using stereo camera setup. In this case, we'd get more accuracy. Also this would be useful in many other stuff like determining the shape of the head etc.

* Try different points for pose estimation, see which one gives the best results

* Try to account for opening the mouth (jaw points are then also affected) etc.

* Sometimes `pose.py` gives wrong rotation values - roll is around 180&deg; instead of 0&deg; which doesn't make any sense. This can be seen particularly well in `live_3D.py`. Need to account for this somehow.

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

As seen in the above photo, we can clearly notice how half closed detected points give a smaller area compared to model ("ground truth"). This can be done on any face angle using the rotation and translation vectors we obtained, thus accounting for all types of head movements. The corners should theoretically be at the same place for both types of points, but this is not achieved because of accuracy of head pose estimation and face model being too generic.

* **TODO:** Fix the way coordinates are stored and optimize `face_model.txt`, as currently it's a very approximate model.

## 3D visualization software

Currently we're using `pyqtgraph`to plot the 3D visualizations. This python library uses OpenGL for 3D object rendering and the application itself is a Qt app. However, there's a known bug (descibred <a href="">here</a>) which basically means that on some devices (like Macs) we can only use 1/4th of the display.

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
