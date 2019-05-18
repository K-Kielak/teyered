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
from teyered.io.image_processing import draw_pose, draw_facial_points, display_video, gray_video, resize_video, write_angles
from teyered.head_pose.pose import estimate_pose, _get_rotation_matrix
from teyered.data_processing.points_extractor import FacialPointsExtractor
from teyered.io.files import load_video

import time


VERTICAL_LINE = "\n-----------------------\n"

VIDEO_PATH = 'video.mov'
FACE_SIZE_RATIO = 0.05


class Live3D():

    def __init__(self):
        print('\nTEYERED: prerecorded 3D')

        print(VERTICAL_LINE)
        print('1. Setting up...')
        # Set up the QT GUI application
        self.application = QtGui.QApplication([])

        # Set up the main window widget and camera
        self.window = gl.GLViewWidget()
        self.window.setWindowTitle('TEYERED: prerecorded 3D')
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
        input("6. 3D visualisation is ready. Press Enter to view...")

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

        # Set up scatter plot and plot initial points
        random_points = np.zeros((10, 3))
        self.face_point_scatter = gl.GLScatterPlotItem()
        self.face_point_scatter.rotate(-90, 1, 0, 0)
        self.face_point_scatter.setData(
            pos = random_points,
            color = (0.5, 0.0, 0.5, 1.0),
            size = 10,
            pxMode = True
        )
        self.window.addItem(self.face_point_scatter)

        # 3D face rotation lines
        random_points_5 = np.zeros((4, 3))
        self.rot_x = gl.GLLinePlotItem()
        self.rot_x.rotate(-90, 1, 0, 0)
        self.rot_x.setData(
            pos = random_points_5,
            color = (0.0, 1.0, 1.0, 1.0),
            width = 10,
            mode = 'line_strip'
        )
        self.window.addItem(self.rot_x)

        self.rot_y = gl.GLLinePlotItem()
        self.rot_y.rotate(-90, 1, 0, 0)
        self.rot_y.setData(
            pos = random_points_5,
            color = (0.0, 1.0, 1.0, 1.0),
            width = 10,
            mode = 'line_strip'
        )
        self.window.addItem(self.rot_y)

        self.rot_z = gl.GLLinePlotItem()
        self.rot_z.rotate(-90, 1, 0, 0)
        self.rot_z.setData(
            pos = random_points_5,
            color = (0.0, 1.0, 1.0, 1.0),
            width = 10,
            mode = 'line_strip'
        )
        self.window.addItem(self.rot_z)

    def _prepare_2D(self):
        camera_model = CameraModel()
        points_extractor = FacialPointsExtractor()

        print(VERTICAL_LINE)
        print('4. Calibrating camera...')
        camera_model.calibrate_custom_parameters(CAMERA_MATRIX, DIST_COEFFS)

        print(VERTICAL_LINE)
        print('5. Loading and analysing video...')
        frames_original = load_video(VIDEO_PATH)
        frames_resized = resize_video(frames_original)
        frames = gray_video(frames_resized)
        self.facial_points_all = points_extractor.extract_facial_points(frames)
        self.r_vectors_all, self.t_vectors_all, self.angles_all, self.camera_world_coord_all = estimate_pose(self.facial_points_all)

        self.max_counter = len(frames)

    def update(self):
        if (self.counter == self.max_counter):
            sys.exit(0)
            return

        # Prepare facial points plot
        facial_points_3D = np.hstack((self.facial_points_all[self.counter], np.zeros((self.facial_points_all[self.counter].shape[0], 1))))

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
        r_matrix, _ = _get_rotation_matrix(self.r_vectors_all[self.counter])
        rotated_x = r_matrix.dot([1,0,0])
        rotated_y = r_matrix.dot([0,1,0])
        rotated_z = r_matrix.dot([0,0,1])

        # Update 3D plots
        self.face_point_scatter.setData(
            pos = facial_points_3D_scaled*FACE_SIZE_RATIO*75,
            color = (1.0, 1.0, 0.0, 1.0),
            size = 10,
            pxMode = True
        )

        self.rot_x.setData(
            pos = np.array([[0,0,0], rotated_x])*10,
            color = (0.0, 1.0, 1.0, 1.0),
            width = 10,
            mode = 'line_strip'
        )

        self.rot_y.setData(
            pos = np.array([[0,0,0], rotated_y])*10,
            color = (0.0, 1.0, 1.0, 1.0),
            width = 10,
            mode = 'line_strip'
        )

        self.rot_z.setData(
            pos = np.array([[0,0,0], rotated_z])*10,
            color = (0.0, 1.0, 0.0, 1.0),
            width = 10,
            mode = 'line_strip'
        )

        # Update counter
        self.counter += 1

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
