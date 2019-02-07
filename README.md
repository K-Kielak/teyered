# Teyered

Machine Learning framework for tiredness detection

### Facial features extraction

We combine CNN facial points detection and Lukas-Kanade optical flow tracker to extract the most accurate representation of facial features at any point in time.

### Normalization of eye points

Eye area depends on various factors that are impossible to control, such as the parameters of the camera, head movement, natural eye shape and closeness to the screen to name a few. To solve this problem, we came up with an algorithm that maps eye points on a universal size grid, which can be then used to compare eyes in different environments.

### Head pose estimation

We use a single monocular RGB camera to accurately track the 3D position and rotations of the face in space. While this, in principle, is a simple PnP problem, we combine facial features extraction, Kalman filter and tuned PnP algorithm to achieve high accuracy.
