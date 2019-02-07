import logging

import cv2
import numpy as np


logger = logging.getLogger(__name__)


class CameraModel:
    """
    Real life monocular RGB camera model and its calibration using 7x6
    chessboard pattern
    """

    def __init__(self):
        self._terminating_criteria = (cv2.TERM_CRITERIA_EPS + \
                                      cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.initialize_camera_data()

    def initialize_camera_data(self):
        # [ (0,0,0), (0,1,0), (1,0,0) ... (6,5,0) ]
        self._objp = np.zeros((6*7,3), np.float32)
        self._objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

        self._obj_points = []  # 3D points in real world space
        self._img_points = []  # 2D points in the image plane

        self._calibrated = False

        logger.debug('Camera data has been cleared successfully')

    def calibrate_camera_squares(self, images, square_size = 1):
        """
        Calibrate camera using chessboard pattern (7x6 from OpenCV)
        :param images: Array of numpy arrays representing images to be used
        for calibration
        :param square_size: Size of a side of a square (in any units)
        """
        if not images:
            logger.warning('You must provide images for calibration')
            return

        # Remove previously stored data (uncalibrate)
        self.initialize_camera_data()

        self._objp *= square_size  # Scale to the units of square size
        self._img_sample = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)

        img_count = 0  # Keep count of images that were used for calibration
        for image in images:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray_image, (7,6), None)

            if ret == True:
                self._obj_points.append(self._objp)
                self._img_points.append(corners)
                img_count += 1

        # Calibration step
        _, self._camera_matrix, self._distortion_coeff, self._rotation_vecs,
        self._translation_vecs = cv2.calibrateCamera(self._obj_points,
                                                     self._img_points,
                                                     self._img_sample.shape[::-1],
                                                     None, None)

        # Calculate optimal matrix
        h,w = self._img_sample.shape[:2]
        self._optimal_camera_matrix,
        self._roi = cv2.getOptimalNewCameraMatrix(self._camera_matrix,
                                                 self._distortion_coeff,
                                                 (w,h), 1, (w,h))

        self._calibration_image_ratio = (img_count / len(images))
        self._calibrated = True
        logger.debug('Camera has been calibrated successfully with ratio' + \
            f'{self._calibration_image_ratio} (images used/all images)')

    def get_calibration_image_ratio(self):
        if not self._calibrated:
            logger.warning('Camera is not calibrated')
            return None
        return self._calibration_image_ratio

    def get_status(self):
        return self._calibrated

    def get_camera_matrix(self):
        if not self._calibrated:
            logger.warning('Camera is not calibrated')
            return None
        return self._camera_matrix

    def get_distortion_coeff(self):
        if not self._calibrated:
            logger.warning('Camera is not calibrated')
            return None
        return self._distortion_coeff

    def get_undistort_image(self, image):
        if not self._calibrated:
            logger.warning('Camera is not calibrated')
            return None

        undistorted_image = cv2.undistort(image, self._camera_matrix,
                                          self._distortion_coeff, None,
                                          self._optimal_camera_matrix)
        x,y,w,h = self._roi
        undistorted_image = undistorted_image[y:y+h, x:x+w]

        return undistorted_image

    def get_reprojection_error(self):
        """
        :return: Mean of reprojection error to estimate how exact the found
        parameters are
        """
        if not self._calibrated:
            logger.warning('Camera is not calibrated')
            return None

        mean_error = 0
        for i in range(0, len(self._obj_points)):
            img_points_prj, _ = cv2.projectPoints(self._obj_points[i],
                                                  self._rotation_vecs[i],
                                                  self._translation_vecs[i],
                                                  self._camera_matrix,
                                                  self._distortion_coeff)
            error = cv2.norm(self._img_points[i], img_points_prj,
                             cv2.NORM_L2) / len(img_points_prj)
            mean_error += error

        return (mean_error/len(self._obj_points))
