import cv2
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import FACE_DATASET_DIR
from utils.file_utils import create_person_directory
from gui.gui_components import show_popup, get_user_input, show_error

class FaceDetector:
    def __init__(self):
        self.dataset_path = FACE_DATASET_DIR
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
    def collect_face_data(self, person_name=None, max_images=1000):
        """Collect face data for a person"""
        if not person_name:
            person_name = get_user_input("Enter the name of the person:")
            
        if not person_name:
            show_error("No name provided!")
            return
            
        person_folder = create_person_directory(person_name, self.dataset_path)
        
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            show_error("Cannot open camera!")
            return
            
        count = 0
        show_popup(f"Starting face data collection for {person_name}. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Ensure frame is in the correct format
            if frame.dtype != 'uint8':
                frame = cv2.convertScaleAbs(frame)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face_filename = person_folder / f"{person_name}_{count}.jpg"
                cv2.imwrite(str(face_filename), face)
                count += 1

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"Images: {count}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Face Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q') or count >= max_images:
                break

        cap.release()
        cv2.destroyAllWindows()
        show_popup(f"Collected {count} images for {person_name}")

def main():
    detector = FaceDetector()
    detector.collect_face_data()

if __name__ == "__main__":
    main()