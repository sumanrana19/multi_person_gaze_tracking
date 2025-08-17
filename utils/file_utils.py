import os
import requests
import logging
from pathlib import Path
from config import SHAPE_PREDICTOR_PATH, SHAPE_PREDICTOR_URL

logger = logging.getLogger(__name__)

def download_shape_predictor_model():
    """Download the shape predictor model if it doesn't exist"""
    if not SHAPE_PREDICTOR_PATH.exists():
        logger.info("Shape predictor model not found. Downloading...")
        try:
            response = requests.get(SHAPE_PREDICTOR_URL, stream=True)
            response.raise_for_status()
            with open(SHAPE_PREDICTOR_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Shape predictor model downloaded to {SHAPE_PREDICTOR_PATH}")
            return True
        except Exception as e:
            logger.error(f"Failed to download shape predictor model: {e}")
            return False
    else:
        logger.info("Shape predictor model already exists")
        return True

def ensure_directories_exist(directories):
    """Ensure all required directories exist"""
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def get_relative_path(target_path, base_path=None):
    """Get relative path from base path to target path"""
    if base_path is None:
        base_path = Path.cwd()
    target_path = Path(target_path)
    base_path = Path(base_path)
    return target_path.relative_to(base_path)

def validate_image_file(file_path):
    """Validate if file is a valid image"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    return Path(file_path).suffix.lower() in valid_extensions

def create_person_directory(person_name, base_dir):
    """Create directory for a person's face images"""
    person_dir = Path(base_dir) / person_name
    person_dir.mkdir(parents=True, exist_ok=True)
    return person_dir
