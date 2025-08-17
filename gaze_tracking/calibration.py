from __future__ import division
import cv2
import numpy as np
from .pupil import Pupil


class Calibration(object):
    """
    This class calibrates the pupil detection algorithm by finding the
    best binarization threshold value for the person and the webcam.
    """

    def __init__(self):
        self.nb_frames = 20
        self.thresholds_left = []
        self.thresholds_right = []

    def is_complete(self):
        """Returns true if the calibration is completed"""
        return len(self.thresholds_left) >= self.nb_frames and len(self.thresholds_right) >= self.nb_frames

    def threshold(self, side):
        """Returns the threshold value for the given eye.

        Argument:
            side: Indicates whether it's the left eye (0) or the right eye (1)
        """
        if side == 0 and self.thresholds_left:
            return int(sum(self.thresholds_left) / len(self.thresholds_left))
        elif side == 1 and self.thresholds_right:
            return int(sum(self.thresholds_right) / len(self.thresholds_right))
        else:
            return 50  # Default threshold

    @staticmethod
    def iris_size(frame):
        """Returns the percentage of space that the iris takes up on
        the surface of the eye.

        Argument:
            frame (numpy.ndarray): Binarized iris frame
        """
        if frame is None or frame.size == 0:
            return 0
            
        # Crop frame to avoid border effects
        height, width = frame.shape[:2]
        if height > 10 and width > 10:
            frame = frame[5:-5, 5:-5]
            height, width = frame.shape[:2]
        
        nb_pixels = height * width
        if nb_pixels == 0:
            return 0
            
        try:
            nb_blacks = nb_pixels - cv2.countNonZero(frame)
            return nb_blacks / nb_pixels
        except cv2.error:
            return 0

    @staticmethod
    def find_best_threshold(eye_frame):
        """Calculates the optimal threshold to binarize the
        frame for the given eye.

        Argument:
            eye_frame (numpy.ndarray): Frame of the eye to be analyzed
        """
        if eye_frame is None or eye_frame.size == 0:
            return 50
            
        average_iris_size = 0.48
        trials = {}

        for threshold in range(5, 100, 5):
            try:
                iris_frame = Pupil.image_processing(eye_frame, threshold)
                if iris_frame is not None and iris_frame.size > 0:
                    trials[threshold] = Calibration.iris_size(iris_frame)
                else:
                    trials[threshold] = 0
            except (cv2.error, ValueError):
                trials[threshold] = 0

        if not trials:
            return 50
            
        # Find threshold closest to average iris size
        best_threshold = min(trials.items(), key=(lambda p: abs(p[1] - average_iris_size)))[0]
        return best_threshold

    def evaluate(self, eye_frame, side):
        """Improves calibration by taking into consideration the
        given image.

        Arguments:
            eye_frame (numpy.ndarray): Frame of the eye
            side: Indicates whether it's the left eye (0) or the right eye (1)
        """
        if eye_frame is None or eye_frame.size == 0:
            return
            
        threshold = self.find_best_threshold(eye_frame)

        if side == 0:
            self.thresholds_left.append(threshold)
        elif side == 1:
            self.thresholds_right.append(threshold)