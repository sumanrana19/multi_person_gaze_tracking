# Face Recognition Module
# This module contains all face recognition functionality

from .face_det import FaceDetector
from .train import FaceTrainer

__version__ = "1.0.0"
__all__ = ["FaceDetector", "FaceTrainer"]