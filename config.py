import os
import sys
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Add project root to Python path
sys.path.append(str(PROJECT_ROOT))

# Directory paths (OS-agnostic)
FACE_DATASET_DIR = PROJECT_ROOT / "face_recognition_module" / "face_dataset"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
TRAINED_MODELS_DIR = PROJECT_ROOT / "gaze_tracking" / "trained_models"

# File paths
FACE_ENCODINGS_PATH = DATA_DIR / "face_encodings.pkl"
ATTENDANCE_PATH = DATA_DIR / "attendance.xlsx"
GAZE_DATA_PATH = DATA_DIR / "gaze_data.xlsx"
SHAPE_PREDICTOR_PATH = TRAINED_MODELS_DIR / "shape_predictor_68_face_landmarks.dat"

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'suman@5142',  # Change this to your MySQL password
    'database': 'dualeye'
}

# Create necessary directories
for directory in [FACE_DATASET_DIR, DATA_DIR, LOGS_DIR, TRAINED_MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Application settings
ATTENDANCE_DURATION = 20  # seconds
GAZE_TRACKING_DURATION = 30  # seconds
VERIFICATION_DURATION = 15  # seconds
MAX_VERIFICATION_ATTEMPTS = 3
FACE_RECOGNITION_TOLERANCE = 0.45
ATTENTION_THRESHOLD = 3  # seconds for non-attentive periods

# Download URL for shape predictor model
SHAPE_PREDICTOR_URL = "https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat"