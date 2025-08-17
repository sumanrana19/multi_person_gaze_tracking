#!/usr/bin/env python3
"""
Quick setup script for Dual-Person Gaze Tracking System
This script helps set up the environment and download required models
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        print(f"✗ Python 3.8+ required, found Python {version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor} detected")
    return True

def setup_virtual_environment():
    """Set up Python virtual environment"""
    if not Path("venv").exists():
        if not run_command("python -m venv venv", "Creating virtual environment"):
            return False
    
    # Activate virtual environment and install requirements
    if sys.platform == "win32":
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    # Upgrade pip
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing Python packages"):
        return False
    
    return True

def setup_directories():
    """Create necessary directories"""
    directories = [
        "face_recognition_module/face_dataset",
        "gaze_tracking/trained_models",
        "data",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    return True

def check_mysql():
    """Check if MySQL is available"""
    try:
        subprocess.run("mysql --version", shell=True, check=True, 
                      capture_output=True, text=True)
        print("✓ MySQL detected")
        return True
    except subprocess.CalledProcessError:
        print("⚠ MySQL not detected. Please install MySQL Server manually.")
        print("  Download from: https://dev.mysql.com/downloads/mysql/")
        return False

def main():
    """Main setup function"""
    print("="*60)
    print("Dual-Person Gaze Tracking System - Quick Setup")
    print("="*60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Set up directories
    print("\nSetting up directories...")
    setup_directories()
    
    # Set up virtual environment
    print("\nSetting up Python environment...")
    if not setup_virtual_environment():
        return False
    
    # Check MySQL
    print("\nChecking MySQL...")
    check_mysql()
    
    print("\n" + "="*60)
    print("Setup Summary:")
    print("="*60)
    print("✓ Python environment configured")
    print("✓ Required packages installed")
    print("✓ Directory structure created")
    print("\nNext steps:")
    print("1. Set up MySQL database (if not already done)")
    print("2. Update database credentials in config.py")
    print("3. Collect face data: python face_recognition_module/face_det.py")
    print("4. Train face recognition: python face_recognition_module/train.py")
    print("5. Run the application: python main.py")
    
    print("\nFor detailed instructions, see README.md")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n✗ Setup failed. Please check the errors above and try again.")
        sys.exit(1)
    print("\n✓ Setup completed successfully!")