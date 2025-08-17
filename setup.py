from setuptools import setup, find_packages

setup(
    name="dual-person-gaze-tracker",
    version="1.0.0",
    description="Dual-Person Smart Attendance & Gaze-Based Attentiveness Monitoring System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.8.0",
        "face_recognition>=1.3.0",
        "face_recognition_models>=0.3.0",
        "dlib>=19.24.2",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "openpyxl>=3.1.0",
        "mysql-connector-python>=8.0.33",
        "Pillow>=9.5.0",
        "imutils>=0.5.4",
        "requests>=2.31.0",
    ],
    entry_points={
        "console_scripts": [
            "dual-gaze-tracker=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)