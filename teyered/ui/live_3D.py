import math
import os
import sys
import time

import cv2
from imutils import face_utils
import imutils
import dlib
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg

from teyered.config import CAMERA_MATRIX, DIST_COEFFS, UNIVERSAL_RESIZE, RED_COLOR_3D, GREEN_COLOR_3D, BLUE_COLOR_3D, YELLOW_COLOR_3D
from teyered.head_pose.camera_model import CameraModel
from teyered.io.image_processing import gray_image, resize_image
from teyered.head_pose.pose import estimate_pose_live, get_rotation_matrix
from teyered.data_processing.points_extractor import FacialPointsExtractor
from teyered.head_pose.face_model_processing import load_face_model, optimize_face_model

VERTICAL_LINE = "\n-----------------------\n"

FACE_SIZE_RATIO = 0.05

RED_COLOR = (1.0, 0.0, 0.0, 1.0)
GREEN_COLOR = (0.0, 1.0, 0.0, 1.0)
BLUE_COLOR = (0.0, 0.0, 1.0, 1.0)
YELLOW_COLOR = (1.0, 1.0, 0.0, 1.0)


class Live3D():

    def __init__(self):
        print(f'{VERTICAL_LINE}\nTEYERED: live 2D')

        print(f'{VERTICAL_LINE}\n1. Setting up...')
        # Set up the QT GUI application
        self.application = QtGui.QApplication([])

        # Set up the main window widget and camera
        self.window = gl.GLViewWidget()
        self.window.setWindowTitle('TEYERED: live 3D')
        self.window.setCameraPosition(distance=30, elevation=5, azimuth=0)
        self.window.setGeometry(0, 110, 1280, 720)
        self.window.show()

        # Setup 3D
        print(f"{VERTICAL_LINE}\n2. Preparing 3D objects...")
        self._prepare_3D()

        # Setup 2D
        print(f"{VERTICAL_LINE}\n3. Preparing 2D objects...")
        self._prepare_2D()

        # Counter for updates
        self.counter = 0

        input(f"{VERTICAL_LINE}\n6. Live 3D is ready. Press Enter to view...")

    def _prepare_3D(self):
        # Background grids for orientation purposes
        gx = gl.GLGridItem()
        gx.rotate(90, 0, 1, 0)
        gx.translate(-10, 0, 0)
        self.window.addItem(gx)
        gy = gl.GLGridItem()
        gy.rotate(90, 1, 0, 0)
        gy.translate(0, 10, 0)
        self.window.addItem(gy)
        gz = gl.GLGridItem()
        gz.translate(0, 0, 10)
        self.window.addItem(gz)
        gx2 = gl.GLGridItem()
        gx2.rotate(90, 0, 1, 0)
        gx2.translate(10, 0, 0)
        self.window.addItem(gx2)
        gy2 = gl.GLGridItem()
        gy2.rotate(90, 1, 0, 0)
        gy2.translate(0, -10, 0)
        self.window.addItem(gy2)

        # Axis display for orientation purposes
        axis = gl.GLAxisItem()
        self.window.addItem(axis)

        random_points = np.zeros((10, 3))

        # Set up scatter plot and plot initial points
        self.face_point_scatter = gl.GLScatterPlotItem()
        self.face_point_scatter.setData(
            pos = random_points,
            color = (0.5, 0.0, 0.5, 1.0),
            size = 10,
            pxMode = True
        )
        self.window.addItem(self.face_point_scatter)

        # 3D face rotation lines
        self.rot_x = gl.GLLinePlotItem()
        self.rot_x.setData(
            pos = random_points,
            color = BLUE_COLOR, #B
            width = 10,
            mode = 'line_strip'
        )
        self.window.addItem(self.rot_x)

        self.rot_y = gl.GLLinePlotItem()
        self.rot_y.setData(
            pos = random_points,
            color = GREEN_COLOR, #G
            width = 10,
            mode = 'line_strip'
        )
        self.window.addItem(self.rot_y)

        self.rot_z = gl.GLLinePlotItem()
        self.rot_z.setData(
            pos = random_points,
            color = RED_COLOR, #R
            width = 10,
            mode = 'line_strip'
        )
        self.window.addItem(self.rot_z)

        # 3D axis
        self.axis_x = gl.GLLinePlotItem()
        self.axis_x.setData(
            pos = np.array([[0,0,0],[1,0,0]]),
            color = BLUE_COLOR, #B
            width = 10,
            mode = 'line_strip'
        )
        self.window.addItem(self.axis_x)

        self.axis_y = gl.GLLinePlotItem()
        self.axis_y.setData(
            pos = np.array([[0,0,0],[0,-1,0]]),
            color = GREEN_COLOR, #G
            width = 10,
            mode = 'line_strip'
        )
        self.window.addItem(self.axis_y)

        self.axis_z = gl.GLLinePlotItem()
        self.axis_z.setData(
            pos = np.array([[0,0,0],[0,0,-1]]),
            color = RED_COLOR, #R
            width = 10,
            mode = 'line_strip'
        )
        self.window.addItem(self.axis_z)

    def _prepare_2D(self):
        camera_model = CameraModel()
        self.points_extractor = FacialPointsExtractor()
        self.cap = cv2.VideoCapture(0)

        print(f'{VERTICAL_LINE}\n4. Calibrating camera...')
        camera_model.calibrate_custom_parameters(CAMERA_MATRIX, DIST_COEFFS)

        print(f'{VERTICAL_LINE}\n5. Loading parameters...')
        self.previous_frame_resized = None
        self.previous_points = None
        self.prev_rvec = None
        self.prev_tvec = None
        self.frame_count = 0

        # Setup model points (currently no optimization)
        model_points_original = load_face_model() # Keep original points for later use
        (self.model_points_optimized, _, model_points_normalized) = optimize_face_model(model_points_original, model_points_original)
        self.model_points = model_points_normalized # Set the actual model points here, must be normalized

    def update(self):
        _, frame = self.cap.read()

        frame_resized = resize_image(frame)
        frame_resized_original = frame_resized # No time to test, better safe than sorry

        # Previous frame is None (either wasnt captured or facial points couldnt be extracted)
        if self.previous_frame_resized is None or self.previous_points is None:
            gray_frame = gray_image(frame_resized)
            facial_points, self.frame_count = self.points_extractor.extract_facial_points_live(gray_frame)

            if facial_points is None:
                self.previous_frame_resized = None
                self.previous_points = None
                self.prev_rvec = None
                self.prev_tvec = None
                self.frame_count = 0
                return

            self.previous_frame_resized = frame_resized_original
            self.previous_points = facial_points
            self.prev_rvec = None
            self.prev_tvec = None

        # Image analysis on current and previous frame
        else:
            gray_frame = gray_image(frame_resized)
            gray_previous_frame = gray_image(self.previous_frame_resized)
            facial_points, self.frame_count = self.points_extractor.extract_facial_points_live(gray_frame, gray_previous_frame, self.previous_points, self.frame_count)

            # Facial points were not detected for some reason on this frame, reset everything
            if facial_points is None:
                self.previous_frame_resized = None
                self.previous_points = None
                self.prev_rvec = None
                self.prev_tvec = None
                self.frame_count = 0
                return

            self.previous_frame_resized = frame_resized_original
            self.previous_points = facial_points

        # Pose estimation
        r_vector, t_vector, angles, camera_world_coord = estimate_pose_live(facial_points, self.model_points_optimized, self.prev_rvec, self.prev_tvec)

        # Print angles
        print(f'yaw: {angles[0][0]}')
        print(f'pitch: {angles[1][0]}')
        print(f'roll: {angles[2][0]}\n')

        # Set previous vectors
        self.prev_rvec = r_vector
        self.prev_tvec = t_vector

        # Prepare facial points plot
        facial_points_3D = np.hstack((self.previous_points, np.zeros((self.previous_points.shape[0], 1))))

        # Shift the points to mean (for changes in left/right of the image)
        facial_points_3D_shifted = facial_points_3D - np.mean(facial_points_3D, axis=0)

        # Scale for being further or closer to the screen
        st_dev = np.std(facial_points_3D, axis=0)
        facial_points_3D_scaled_x = facial_points_3D_shifted[:,0] / st_dev[0]
        facial_points_3D_scaled_y = facial_points_3D_shifted[:,1] / st_dev[1]
        facial_points_3D_scaled_z =  np.zeros((facial_points_3D_shifted.shape[0],))

        facial_points_3D_scaled = []
        for i in range(0, facial_points_3D_scaled_x.shape[0]):
            facial_points_3D_scaled.append([facial_points_3D_scaled_x[i], facial_points_3D_scaled_y[i], facial_points_3D_scaled_z[i]])
        facial_points_3D_scaled = np.array(facial_points_3D_scaled)

        # This is a mistake with rotation matrix, should be returned
        r_matrix, _ = get_rotation_matrix(self.prev_rvec)
        rotated_x = r_matrix.dot([1,0,0])
        rotated_y = r_matrix.dot([0,-1,0])
        rotated_z = r_matrix.dot([0,0,-1])

        # Update 3D plots
        self.face_point_scatter.setData(
            pos = facial_points_3D_scaled*FACE_SIZE_RATIO*75,
            color = YELLOW_COLOR,
            size = 10,
            pxMode = True
        )

        self.rot_x.setData(
            pos = np.array([[0,0,0], rotated_x])*10,
            color = BLUE_COLOR,
            width = 10,
            mode = 'line_strip'
        )

        self.rot_y.setData(
            pos = np.array([[0,0,0], rotated_y])*10,
            color = GREEN_COLOR, 
            width = 10,
            mode = 'line_strip'
        )

        self.rot_z.setData(
            pos = np.array([[0,0,0], rotated_z])*10,
            color = RED_COLOR,
            width = 10,
            mode = 'line_strip'
        )

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def animation(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(50)
        self.start()
        self.update()

if __name__ == '__main__':
    live = Live3D()
    live.animation()
    print(f'{VERTICAL_LINE}\n8. Quitting...\n{VERTICAL_LINE}')
