import os
import sys

import cv2
from imutils import face_utils
import imutils
import dlib

import numpy as np

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg

import math

from teyered.config import CAMERA_MATRIX, DIST_COEFFS, UNIVERSAL_RESIZE
from teyered.head_pose.camera_model import CameraModel
from teyered.io.image_processing import draw_pose_frame, draw_facial_points_frame, gray_image, resize_image, write_angles_frame
from teyered.head_pose.pose import estimate_pose_live, _prepare_face_model, _get_rotation_matrix, _prepare_original_face_model, _choose_pose_points
from teyered.data_processing.points_extractor import FacialPointsExtractor
from teyered.io.files import load_video, load_image

from teyered.head_pose.face_model_processing import optimize_face_model, get_ground_truth

import time


VERTICAL_LINE = "\n-----------------------\n"

FACE_SIZE_RATIO = 0.05

cap = cv2.VideoCapture(0)

GROUND_TRUTH_FRAME = 'ground_truth/frame.jpg'


class Live3D():

    def __init__(self):
        print('\nTEYERED: live 3D')

        print(VERTICAL_LINE)
        print('1. Setting up...')
        # Set up the QT GUI application
        self.application = QtGui.QApplication([])

        # Set up the main window widget and camera
        self.window = gl.GLViewWidget()
        self.window.setWindowTitle('TEYERED: live 3D')
        self.window.setCameraPosition(distance=30, elevation=5, azimuth=0)
        self.window.setGeometry(0, 110, 1280, 720)
        self.window.show()

        # Setup 3D
        print(VERTICAL_LINE)
        print("2. Preparing 3D objects...")
        self._prepare_3D()

        # Setup 2D
        print(VERTICAL_LINE)
        print("3. Preparing 2D objects...")
        self._prepare_2D()

        # Counter for updates
        self.counter = 0

        print(VERTICAL_LINE)
        input("6. Live 3D is ready. Press Enter to view...")

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
        gz.translate(0, 0, -10)
        self.window.addItem(gz)
        gx2 = gl.GLGridItem()
        gx2.rotate(90, 0, 1, 0)
        gx2.translate(10, 0, 0)
        self.window.addItem(gx2)
        gz2 = gl.GLGridItem()
        gz2.translate(0, 0, 10)
        self.window.addItem(gz2)

        # Axis display for orientation purposes
        axis = gl.GLAxisItem()
        self.window.addItem(axis)

        # Set up model scatter plot and plot initial points
        random_points = np.zeros((10, 3))
        self.model_points_scatter = gl.GLScatterPlotItem()
        self.model_points_scatter.setData(
            pos = random_points,
            color = (0.5, 0.0, 0.5, 1.0),
            size = 10,
            pxMode = True
        )
        self.window.addItem(self.model_points_scatter)

        # Set up optimized model points plot
        self.o_model_points_scatter = gl.GLScatterPlotItem()
        self.o_model_points_scatter.setData(
            pos = random_points,
            color = (0.0, 0.5, 0.0, 1.0),
            size = 10,
            pxMode = True
        )
        self.window.addItem(self.o_model_points_scatter)

        # Set up original unscaled untranslated model points plot
        self.to_model_points_scatter = gl.GLScatterPlotItem()
        self.to_model_points_scatter.setData(
            pos = random_points,
            color = (0.0, 0.5, 0.0, 1.0),
            size = 10,
            pxMode = True
        )
        self.window.addItem(self.to_model_points_scatter)

        # Set up face points plot
        self.facial_points_scatter = gl.GLScatterPlotItem()
        self.facial_points_scatter.setData(
            pos = random_points,
            color = (0.0, 0.5, 0.0, 1.0),
            size = 10,
            pxMode = True
        )
        self.window.addItem(self.facial_points_scatter)

        # Set up old 3D model plot
        self.old_face_model_scatter = gl.GLScatterPlotItem()
        self.old_face_model_scatter.setData(
            pos = random_points,
            color = (0.0, 0.5, 0.0, 1.0),
            size = 10,
            pxMode = True
        )
        self.window.addItem(self.old_face_model_scatter)

        # Axis
        self.x_axis = gl.GLLinePlotItem()
        self.x_axis.setData(
            pos = np.array([[0,0,0], [1,0,0]])*10,
            color = (1.0, 0.0, 0.0, 1.0), # R
            width = 15,
            mode = 'line_strip'
        )
        self.window.addItem(self.x_axis)
        self.y_axis = gl.GLLinePlotItem()
        self.y_axis.setData(
            pos = np.array([[0,0,0], [0,1,0]])*10,
            color = (0.0, 1.0, 0.0, 1.0), # G
            width = 15,
            mode = 'line_strip'
        )
        self.window.addItem(self.y_axis)
        self.z_axis = gl.GLLinePlotItem()
        self.z_axis.setData(
            pos = np.array([[0,0,0], [0,0,1]])*10,
            color = (0.0, 0.0, 1.0, 1.0), # B
            width = 15,
            mode = 'line_strip'
        )
        self.window.addItem(self.z_axis)

    def _prepare_2D(self):
        camera_model = CameraModel()
        self.points_extractor = FacialPointsExtractor()

        print(VERTICAL_LINE)
        print('4. Calibrating camera...')
        camera_model.calibrate_custom_parameters(CAMERA_MATRIX, DIST_COEFFS)

        print(VERTICAL_LINE)
        print('5. Loading parameters...')
        self.previous_frame_resized = None
        self.previous_points = None
        self.prev_rvec = None
        self.prev_tvec = None
        self.frame_count = 0
        self.cap = cv2.VideoCapture(0)

        o_model_points_all = _prepare_original_face_model()
        ground_truth = get_ground_truth(load_image(GROUND_TRUTH_FRAME), self.points_extractor)
        (model_points_all, facial_points_z, model_points_z) = optimize_face_model(ground_truth, o_model_points_all)

        self.model_points = _choose_pose_points(model_points_all)

        """
        self.model_points_scatter.setData(
            pos = model_points_all*2,
            color = (0.5, 0.0, 0.0, 1.0), #R
            size = 10,
            pxMode = True
        )
        """
        """
        self.o_model_points_scatter.setData(
            pos = model_points_z*2,
            color = (0.0, 0.5, 0.0, 1.0), #G
            size = 10,
            pxMode = True
        )
        """
        """
        self.to_model_points_scatter.setData(
            pos = o_model_points_all*0.025,
            color = (0.5, 0, 0.5, 1.0), #BG
            size = 10,
            pxMode = True
        )
        """
        """
        self.facial_points_scatter.setData(
            pos = facial_points_z*2,
            color = (0.5, 0.0, 0.0, 1.0), #BG
            size = 10,
            pxMode = True
        )
        """ 
        """
        old_model = np.array([[0.0,0.0,0.0],[0.0,-330.0, -65.0],[-225.0,170.0,-135.0], [225.0,170.0,-135.0],[-150.0,-150.0,-125.0],[150.0,-150.0,-125.0]])
        self.old_face_model_scatter.setData(
            pos = np.array([old_model[i] for i in [0,1,2,4]])*0.01,
            color = (0.0, 0.5, 0.5, 1.0),
            size = 10,
            pxMode = True
        )
        """


    def update(self):
        _, frame = cap.read()

        frame_resized = resize_image(frame)

        # Previous frame is None (either wasnt captured or facial points couldnt be extracted)
        if self.previous_frame_resized is None:
            gray_frame = gray_image(frame_resized)
            facial_points, count = self.points_extractor.extract_facial_points_live(None, None, gray_frame, 0)

            if facial_points is None:
                self.previous_frame_resized = None
                self.previous_points = None
                self.prev_rvec = None
                self.prev_tvec = None
                self.frame_count = 0
                return

            self.previous_frame_resized = frame_resized
            self.previous_points = facial_points
            self.prev_rvec = None
            self.prev_tvec = None
            self.frame_count = 1

        # Image analysis on current and previous frame
        else:
            gray_frame = gray_image(frame_resized)
            gray_previous_frame = gray_image(self.previous_frame_resized)
            facial_points, count = self.points_extractor.extract_facial_points_live(gray_previous_frame, self.previous_points, gray_frame, self.frame_count)

            # Facial points were not detected for some reason on this frame, reset everything
            if facial_points is None:
                self.previous_frame_resized = None
                self.previous_points = None
                self.prev_rvec = None
                self.prev_tvec = None
                self.frame_count = 0
                return

            self.previous_frame_resized = frame_resized
            self.previous_points = facial_points
            self.frame_count = count

        # Pose estimation
        r_vector, t_vector, angles, camera_world_coord = estimate_pose_live(facial_points, self.prev_rvec, self.prev_tvec, self.model_points)

        # Set previous vectors
        self.prev_rvec = r_vector
        self.prev_tvec = t_vector

        # Prepare facial points plot
        facial_points_3D = np.hstack((self.previous_points, np.zeros((self.previous_points.shape[0], 1))))

        # Shift the points to mean (for changes in left/right of the image)
        mean = np.mean(facial_points_3D, axis=0)
        facial_points_3D_shifted = facial_points_3D - mean

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
        r_matrix, _ = _get_rotation_matrix(self.prev_rvec)
        rotated_x = r_matrix.dot([1,0,0])
        rotated_y = r_matrix.dot([0,1,0])
        rotated_z = r_matrix.dot([0,0,1])

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
