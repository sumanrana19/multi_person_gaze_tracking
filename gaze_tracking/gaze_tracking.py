from __future__ import division
import os
import cv2
import dlib
import numpy as np
from pathlib import Path
from .eye import Eye
from .calibration import Calibration
from config import SHAPE_PREDICTOR_PATH
from utils.file_utils import download_shape_predictor_model

class GazeTracking(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()

        # _face_detector is used to detect faces
        self._face_detector = dlib.get_frontal_face_detector()

        # Download shape predictor model if needed
        if not download_shape_predictor_model():
            raise FileNotFoundError("Could not download shape predictor model")

        # _predictor is used to get facial landmarks of a given face
        try:
            self._predictor = dlib.shape_predictor(str(SHAPE_PREDICTOR_PATH))
        except RuntimeError as e:
            raise RuntimeError(f"Could not load shape predictor model: {e}")

    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except (AttributeError, TypeError):
            return False

    def _analyze(self):
        """Detects the face and initialize Eye objects"""
        if self.frame is None:
            return
            
        # Ensure frame is grayscale and 8-bit
        if len(self.frame.shape) == 3:
            frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        else:
            frame = self.frame.copy()
            
        # Ensure 8-bit format to avoid the 8-bit gray error
        if frame.dtype != np.uint8:
            frame = cv2.convertScaleAbs(frame)
            
        faces = self._face_detector(frame)

        try:
            landmarks = self._predictor(frame, faces[0])
            self.eye_left = Eye(frame, landmarks, 0, self.calibration)
            self.eye_right = Eye(frame, landmarks, 1, self.calibration)
        except (IndexError, RuntimeError):
            self.eye_left = None
            self.eye_right = None

    def refresh(self, frame):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        if frame is not None and frame.size > 0:
            self.frame = frame
            self._analyze()

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)
        return None

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)
        return None

    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2
        return None

    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2
        return None

    def is_right(self):
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.35
        return False

    def is_left(self):
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.65
        return False

    def is_center(self):
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            return not self.is_right() and not self.is_left()
        return False

    def is_blinking(self):
        """Returns true if the user closes his eyes"""
        if self.eye_left is not None and self.eye_right is not None:
            if self.eye_left.blinking is not None and self.eye_right.blinking is not None:
                blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
                return blinking_ratio > 3.8
        return False

    def annotated_frame(self):
        """Returns the main frame with pupils highlighted"""
        if self.frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
            
        frame = self.frame.copy()
        
        # Ensure frame is in BGR format for annotation
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        if self.pupils_located:
            color = (0, 255, 0)
            left_coords = self.pupil_left_coords()
            right_coords = self.pupil_right_coords()
            
            if left_coords:
                x_left, y_left = left_coords
                cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
                cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
                
            if right_coords:
                x_right, y_right = right_coords
                cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
                cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)

        return frame