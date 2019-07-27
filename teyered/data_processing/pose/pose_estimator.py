import logging

import cv2
import numpy as np

from teyered.config import JAW_COORDINATES, NOSE_COORDINATES, CAMERA_MATRIX
from teyered.config import DIST_COEFFS
from teyered.data_processing.projection import project_points
from teyered.data_processing.projection import calculate_reprojection_error


logger = logging.getLogger(__name__)


class PoseEstimator:

    def __init__(self, model_points, camera_matrix=CAMERA_MATRIX,
                 dist_coeffs=DIST_COEFFS):
        self._model_points_pose = self.choose_pose_points(model_points)
        self._camera_matrix = camera_matrix
        self._dist_coeffs = dist_coeffs

        # Previous batch information
        self._prev_rvec = None
        self._prev_tvec = None

    def reset(self):
        self._prev_rvec = None
        self._prev_tvec = None

    def get_prev_rvec(self):
        return self._prev_rvec

    def get_prev_tvec(self):
        return self._prev_tvec

    def get_model_points_pose(self):
        return self._model_points_pose

    def choose_pose_points(self, facial_points):
        """
        Choose facial points which will be used to solve PnP
        :param facial_points: Facial points adhering to face model requirements
        :return: np.ndarray of tuples for specific facial points
        """
        if facial_points.shape[0] != 68:
            raise ValueError('facial_points doesnt have the correct shape')

        points_jaw = facial_points[slice(*JAW_COORDINATES)]
        points_nose = facial_points[slice(*NOSE_COORDINATES)]
        points_r_eye = [facial_points[i] for i in [36, 39]]  # Corners
        points_l_eye = [facial_points[i] for i in [42, 45]]  # Corners
        points_mouth = [facial_points[i] for i in [48, 54]]  # Corners

        points = np.concatenate((points_jaw, points_nose, points_r_eye,
                                 points_l_eye, points_mouth))

        return np.array([tuple(p) for p in points], dtype='double')

    def _get_euler_angles(self, rotation_matrix, translation_vector):
        """
        Get euler angles from rotation matrix and translation vector (XYZ)
        TODO Check the order of Euler angles
        :param rotation_matrix: Rotation matrix from get_rotation_matrix()
        :param translation_vector: Translation vector from solve_pnp()
        :return: Yaw, pitch and roll angles in this specific order
        """
        extrinsic_parameters = np.hstack((rotation_matrix, translation_vector))
        projection_matrix = self._camera_matrix.dot(extrinsic_parameters)
        euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)[-1]

        yaw = euler_angles[1]
        pitch = euler_angles[0]
        roll = euler_angles[2]

        return yaw, pitch, roll

    def _get_rotation_matrix(self, rotation_vector):
        """
        Convert axis and angle of rotation representation to rotation matrix
        :param rotation_vector: Rotation vector from solve_pnp()
        :return: Rotation matrix
        """
        return cv2.Rodrigues(rotation_vector)

    def _solve_pnp(self, image_points):
        """
        Calculate rotation and translation vectors by solving PnP
        TODO Use different pose estimation algorithms to see which performs
        most accurately
        :param image_points: Image points at this specific frame
        :return: np.ndarray of rotation and translation vectors for the frame
        """
        if image_points is None:
            raise ValueError('image_points cannot be None')

        if self._prev_rvec is None or self._prev_tvec is None:
            _, r_vector, t_vector = cv2.solvePnP(self._model_points_pose,
                                                 image_points,
                                                 self._camera_matrix,
                                                 self._dist_coeffs,
                                                 flags=cv2.cv2.SOLVEPNP_ITERATIVE)
        else:
            _, r_vector, t_vector = cv2.solvePnP(self._model_points_pose,
                                                 image_points,
                                                 self._camera_matrix,
                                                 self._dist_coeffs,
                                                 rvec=self._prev_rvec,
                                                 tvec=self._prev_tvec,
                                                 useExtrinsicGuess=True,
                                                 flags=cv2.cv2.SOLVEPNP_ITERATIVE)
        return r_vector, t_vector

    def _get_camera_world_coord(self, rotation_matrix, t_vector):
        """
        Use object's rotation matrix and translation vector to calculate
        camera's position in world coordinates
        :param rotation_matrix: Rotation matrix from get_rotation_matrix()
        :param t_vector: Translation vector from solve_pnp()
        :return: np.ndarray of camera's position in world coordinates
        """
        camera_pose_world = -np.matrix(rotation_matrix).T * np.matrix(t_vector)
        return camera_pose_world.reshape(1, -1)

    def calculate_pose_reprojection_error(self, facial_points_all,
                                          r_vectors_all, t_vectors_all):
        """
        Calculate pose reprojection error to get the accuracy
        :param facial_points_all: Scaled facial points coordinates for all
        frames
        :param r_vectors_all: Rotation vectors for all frames
        :param t_vectors_all: Translation vectors for all frames
        :return: np.ndarray of float values of the reprojection error for each
        frame
        """
        if facial_points_all.shape[0] != r_vectors_all.shape[0]:
            raise ValueError('facial_points_all and r_vectors_all must have the '
                             'same array length')

        model_points_pose_projected_all = project_points(self._model_points_pose,
                                                         r_vectors_all,
                                                         t_vectors_all,
                                                         self._camera_matrix,
                                                         self._dist_coeffs)
        errors = calculate_reprojection_error(facial_points_all, 
                                              model_points_pose_projected_all)
        return errors

    def estimate_pose(self, facial_points_all):
        """
        Estimate 3D pose of an object in camera coordinates from given
        facial points
        :param facial_points_all: Scaled facial points coordinates for all
        frames
        :return: np.ndarray of rotation and translation vectors,
        euler angles and camera position in world coordinates for every
        frame as np.ndarray
        """
        if facial_points_all.size < 1:
            raise ValueError('Facial points must be provided for at '
                             'least one frame')

        # Information to be returned
        r_vectors_all = []
        t_vectors_all = []
        angles_all = []
        camera_world_coord_all = []
        facial_points_pose_all = []

        for facial_points in facial_points_all:

            # No facial points detected for the frame, skip and go to next
            if facial_points is None:
                r_vectors_all.append(None)
                t_vectors_all.append(None)
                angles_all.append(None)
                camera_world_coord_all.append(None)
                self._prev_rvec = None
                self._prev_tvec = None
                continue

            facial_points_pose = self.choose_pose_points(facial_points)
            facial_points_pose_all.append(np.copy(facial_points_pose))
            r_vector, t_vector = self._solve_pnp(facial_points_pose)
            rotation_matrix, _ = self._get_rotation_matrix(r_vector)
            yaw, pitch, roll = self._get_euler_angles(rotation_matrix,
                                                      t_vector)
            camera_world_coord = self._get_camera_world_coord(rotation_matrix,
                                                              t_vector)
            

            r_vectors_all.append(np.copy(r_vector))
            t_vectors_all.append(np.copy(t_vector))
            angles_all.append(np.array([yaw, pitch, roll]))
            camera_world_coord_all.append(np.copy(camera_world_coord))

            self._prev_rvec = np.copy(r_vector)
            self._prev_tvec = np.copy(t_vector)

        logger.debug('Pose estimation has finished successfully')
        return (np.array(r_vectors_all), np.array(t_vectors_all),
                np.array(angles_all), np.array(camera_world_coord_all))
