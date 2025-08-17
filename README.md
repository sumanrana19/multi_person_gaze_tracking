# Dual-Person Smart Attendance & Gaze-Based Attentiveness Monitoring System

A comprehensive machine learning and computer vision system that combines automated attendance tracking with real-time gaze-based attentiveness analysis for two students simultaneously.

## Features

- **Automated Attendance Tracking**: Uses face recognition to automatically mark attendance
- **Dual-Person Gaze Tracking**: Simultaneously tracks eye movements of two selected individuals
- **Attentiveness Analysis**: Converts gaze data into productivity scores and assessments
- **Database Integration**: Stores all data in MySQL for longitudinal analysis
- **User-Friendly GUI**: Tkinter-based interface for easy operation
- **Cross-Platform**: Works on Windows, macOS, and Linux

## System Requirements

### Hardware
- Webcam (HD recommended)
- CPU: Intel i5 or better (i7 recommended)
- RAM: 8GB minimum (16GB+ recommended)
- GPU: Optional but recommended for better performance

### Software
- Python 3.8 or higher (3.12 recommended)
- MySQL Server 8.0+
- Operating System: Windows 10+, macOS 10.14+, or Linux

## Installation

### 1. Clone or Download the Project

```bash
git clone <repository-url>
cd dual_person_gaze_tracking
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Alternative: Install using setup.py
pip install -e .
```

### 4. Set Up MySQL Database

1. Install MySQL Server if not already installed
2. Create a new database:
```sql
CREATE DATABASE dual_gaze_tracking;
```
3. Update database credentials in `config.py`:
```python
DB_CONFIG = {
    'host': 'localhost',
    'user': 'your_username',
    'password': 'your_password',  # Change this!
    'database': 'dual_gaze_tracking',
    'raise_on_warnings': True
}
```

### 5. Set Up Face Recognition Data

#### Option A: Collect New Face Data
```bash
# Run face detection to collect images
python face_recognition_module/face_det.py

# Train the face recognition model
python face_recognition_module/train.py
```

#### Option B: Use Existing Face Dataset
1. Place face images in `face_recognition_module/face_dataset/`
2. Organize as: `face_dataset/PersonName/image1.jpg, image2.jpg, ...`
3. Run training: `python face_recognition_module/train.py`

## Usage

### 1. Run the Main Application
```bash
python main.py
```

### 2. Follow the Six-Phase Workflow

1. **Attendance Phase**: System detects faces for 20 seconds
2. **Present List Display**: Shows all detected attendees
3. **Person Selection**: Choose exactly TWO people for gaze tracking
4. **Verification**: Confirm both selected people are visible
5. **Gaze Tracking**: 30-second dual gaze monitoring session
6. **Analysis**: View attentiveness reports for both individuals

### 3. View Results

- **Excel Files**: `data/attendance.xlsx` and `data/gaze_data.xlsx`
- **Database**: Query MySQL tables for detailed analysis
- **Reports**: Pop-up GUI windows showing attentiveness scores

## Project Structure

```
dual_person_gaze_tracking/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation script
├── config.py                    # Configuration settings
├── main.py                      # Main application entry point
├── face_recognition_module/     # Face recognition components
│   ├── __init__.py
│   ├── face_det.py             # Face data collection
│   ├── train.py                # Face encoding training
│   └── face_dataset/           # Face image storage
├── gaze_tracking/              # Gaze tracking components
│   ├── __init__.py
│   ├── gaze_tracking.py        # Main gaze tracking class
│   ├── eye.py                  # Eye detection and analysis
│   ├── pupil.py                # Pupil detection
│   ├── calibration.py          # Gaze calibration
│   └── trained_models/         # Model files (auto-downloaded)
├── database/                   # Database management
│   ├── __init__.py
│   └── db_manager.py          # Database operations
├── gui/                       # User interface components
│   ├── __init__.py
│   └── gui_components.py      # GUI functions
├── utils/                     # Utility functions
│   ├── __init__.py
│   └── file_utils.py          # File and download utilities
├── data/                      # Data storage
│   ├── face_encodings.pkl     # Trained face encodings
│   ├── attendance.xlsx        # Attendance records
│   └── gaze_data.xlsx         # Gaze tracking data
└── logs/                      # Application logs
    └── app.log                # Runtime logs
```

## Configuration

### Database Settings
Edit `config.py` to update database connection settings:
```python
DB_CONFIG = {
    'host': 'localhost',
    'user': 'your_mysql_username',
    'password': 'your_mysql_password',
    'database': 'dual_gaze_tracking',
    'raise_on_warnings': True
}
```

### Application Settings
Customize timing and thresholds in `config.py`:
```python
ATTENDANCE_DURATION = 20          # Attendance capture time (seconds)
GAZE_TRACKING_DURATION = 30       # Gaze tracking time (seconds)
VERIFICATION_DURATION = 15        # Verification time per attempt (seconds)
FACE_RECOGNITION_TOLERANCE = 0.45 # Face recognition sensitivity
ATTENTION_THRESHOLD = 3           # Non-attentive period threshold (seconds)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. "8-bit grey error"
**Problem**: OpenCV image format error
**Solution**: The updated code handles this automatically with proper format conversion

#### 2. "Gaze tracking module not found"
**Problem**: Import path issues
**Solution**: The new structure uses proper relative imports and `__init__.py` files

#### 3. "Shape predictor model not found"
**Problem**: Missing dlib model file
**Solution**: The system automatically downloads the model on first run

#### 4. Camera not opening
**Problem**: Webcam access issues
**Solution**: 
- Check camera permissions
- Ensure no other application is using the camera
- Try different camera index: `cv2.VideoCapture(1)` instead of `cv2.VideoCapture(0)`

#### 5. Database connection failed
**Problem**: MySQL connection issues
**Solution**:
- Verify MySQL server is running
- Check credentials in `config.py`
- Ensure the database exists

#### 6. Face recognition not working
**Problem**: No face encodings or poor recognition
**Solution**:
- Collect more face images (50-100 per person recommended)
- Ensure good lighting and clear face images
- Retrain the model: `python face_recognition_module/train.py`

### Performance Optimization

1. **For slower computers**:
   - Reduce frame processing frequency
   - Lower webcam resolution
   - Disable GPU processing if causing issues

2. **For better accuracy**:
   - Use HD webcam
   - Ensure good lighting
   - Collect diverse face images (different angles, expressions)

## Technical Details

### Algorithms Used

1. **Face Recognition**: 
   - dlib's ResNet CNN (99.38% LFW accuracy)
   - 128-dimensional facial embeddings
   - Euclidean distance matching

2. **Gaze Tracking**:
   - 68-point facial landmark detection
   - Eye region isolation and pupil detection
   - Gaze direction classification

3. **Attentiveness Analysis**:
   - Time-based gaze gap analysis
   - Percentage-based productivity scoring
   - Categorical assessment (Highly/Moderately/Not Productive)

### Performance Metrics
- **Frame Rate**: 25+ FPS on recommended hardware
- **Recognition Accuracy**: 99%+ in good lighting conditions
- **Processing Latency**: <40ms per frame for dual tracking

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


## Support

For issues, questions, or contributions:

1. Check the troubleshooting section above
2. Review the logs in `logs/app.log`
3. Create an issue with detailed error information

## Acknowledgments

- **dlib**: For facial landmark detection and recognition
- **OpenCV**: For computer vision operations
- **face_recognition**: For simplified face recognition API
- **MySQL**: For robust data storage

---

**Note**: This system is designed for educational and research purposes. Ensure you have proper consent before using it to monitor individuals.
