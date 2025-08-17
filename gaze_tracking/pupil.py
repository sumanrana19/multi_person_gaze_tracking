import numpy as np
import cv2


class Pupil(object):
    """
    This class detects the iris of an eye and estimates
    the position of the pupil
    """

    def __init__(self, eye_frame, threshold):
        self.iris_frame = None
        self.threshold = threshold
        self.x = None
        self.y = None

        self.detect_iris(eye_frame)

    @staticmethod
    def image_processing(eye_frame, threshold):
        """Performs operations on the eye frame to isolate the iris

        Arguments:
            eye_frame (numpy.ndarray): Frame containing an eye and nothing else
            threshold (int): Threshold value used to binarize the eye frame

        Returns:
            A frame with a single element representing the iris
        """
        if eye_frame is None or eye_frame.size == 0:
            return np.zeros((20, 20), dtype=np.uint8)
            
        kernel = np.ones((3, 3), np.uint8)
        
        # Ensure frame is 8-bit grayscale
        if len(eye_frame.shape) == 3:
            new_frame = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
        else:
            new_frame = eye_frame.copy()
            
        # Ensure 8-bit format
        if new_frame.dtype != np.uint8:
            new_frame = cv2.convertScaleAbs(new_frame)
            
        try:
            new_frame = cv2.bilateralFilter(new_frame, 10, 15, 15)
            new_frame = cv2.erode(new_frame, kernel, iterations=3)
            new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)[1]
        except cv2.error:
            # Fallback if processing fails
            new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)[1]

        return new_frame

    def detect_iris(self, eye_frame):
        """Detects the iris and estimates the position of the iris by
        calculating the centroid.

        Arguments:
            eye_frame (numpy.ndarray): Frame containing an eye and nothing else
        """
        if eye_frame is None or eye_frame.size == 0:
            return
            
        self.iris_frame = self.image_processing(eye_frame, self.threshold)

        try:
            contours, _ = cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
            
            if contours:
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                
                # Try to find a suitable contour
                for contour in contours:
                    if cv2.contourArea(contour) > 10:  # Minimum area threshold
                        try:
                            moments = cv2.moments(contour)
                            if moments['m00'] != 0:
                                self.x = int(moments['m10'] / moments['m00'])
                                self.y = int(moments['m01'] / moments['m00'])
                                break
                        except (ZeroDivisionError, ValueError):
                            continue
        except (cv2.error, IndexError, ValueError):
            # If contour detection fails, set default values
            pass