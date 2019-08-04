from collections import namedtuple

import numpy as np

from teyered.data_processing.eyes_processing import calculate_eye_closedness
from teyered.data_processing.facial_points_extractor import FacialPointsExtractor
from teyered.data_processing.image_processing import gray_video, resize_video
from teyered.data_processing.pose.face_model_processing import load_face_model
from teyered.data_processing.pose.pose_estimator import PoseEstimator


# Defines outputs of the processor, fields can be added without checking
# with the code that already uses it. Incompatibility may occur when deleting
# some fields (other code may depend on them).
ProcessingOutput = namedtuple('AnalysisOutput', ['pose_reprojection_err',
                                                 'left_eye_closedness',
                                                 'right_eye_closedness'])


class TeyeredProcessor:

    def __init__(self):
        self._pts_extractor = FacialPointsExtractor()
        self._model_points = load_face_model()
        self._pose_estimator = PoseEstimator(self._model_points)

    def process(self, frames):
        """
        Given frames batch, fully processes it and returns extracted data.
        :param frames: list of frames, should be in chronological order, for
            best results with as small time in between each frame as possible.
        :return: Output of the processing, as specified by the ProcessingOutput
            namedtuple.
        """
        frames = np.array(frames)
        frames = gray_video(resize_video(frames))
        facial_pts = self._pts_extractor.extract_facial_points(frames)
        r_vecs, t_vecs, angles, camera_wcs, pose_errs = \
            self._pose_estimator.estimate_pose(facial_pts)
        closed_left, closed_right = calculate_eye_closedness(facial_pts,
                                                             self._model_points,
                                                             r_vecs,
                                                             t_vecs)
        return ProcessingOutput(pose_errs, closed_left, closed_right)

    def reset(self):
        """
        Resets the state of all processors so the results do not depend
        on previous frames. Useful when reusing the same object for frames
        coming from the new source (e.g. different video).
        """
        self._pts_extractor.reset()
        self._pose_estimator.reset()
