import os
import sys
import pickle
import face_recognition
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import FACE_DATASET_DIR, FACE_ENCODINGS_PATH
from gui.gui_components import show_popup, show_error

class FaceTrainer:
    def __init__(self):
        self.dataset_path = FACE_DATASET_DIR
        self.encodings_path = FACE_ENCODINGS_PATH
        
    def train_face_encodings(self):
        """Train face encodings from dataset"""
        known_encodings = []
        known_names = []
        
        if not self.dataset_path.exists():
            show_error(f"Dataset directory not found: {self.dataset_path}")
            return False
            
        processed_count = 0
        total_images = 0
        
        # Count total images first
        for person_name in os.listdir(self.dataset_path):
            person_folder = self.dataset_path / person_name
            if person_folder.is_dir():
                total_images += len([f for f in os.listdir(person_folder) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if total_images == 0:
            show_error("No images found in dataset directory!")
            return False
            
        print(f"Found {total_images} images to process...")
        
        for person_name in os.listdir(self.dataset_path):
            person_folder = self.dataset_path / person_name
            if person_folder.is_dir():
                print(f"Processing images for {person_name}...")
                person_encodings = 0
                
                for image_name in os.listdir(person_folder):
                    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                        
                    image_path = person_folder / image_name
                    try:
                        image = face_recognition.load_image_file(str(image_path))
                        face_encodings = face_recognition.face_encodings(image)
                        
                        if len(face_encodings) > 0:
                            known_encodings.append(face_encodings[0])
                            known_names.append(person_name)
                            person_encodings += 1
                        else:
                            print(f"No face found in {image_path}")
                            
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
                    
                    processed_count += 1
                    if processed_count % 10 == 0:
                        print(f"Processed {processed_count}/{total_images} images...")
                
                print(f"Created {person_encodings} encodings for {person_name}")
        
        if len(known_encodings) == 0:
            show_error("No face encodings were created!")
            return False
            
        # Save encodings to file
        try:
            with open(self.encodings_path, "wb") as f:
                pickle.dump((known_encodings, known_names), f)
            
            show_popup(f"Training complete! Created {len(known_encodings)} face encodings. Saved to {self.encodings_path.name}")
            return True
            
        except Exception as e:
            show_error(f"Error saving encodings: {e}")
            return False
    
    def load_encodings(self):
        """Load existing face encodings"""
        if not self.encodings_path.exists():
            return None, None
            
        try:
            with open(self.encodings_path, "rb") as f:
                known_encodings, known_names = pickle.load(f)
            return known_encodings, known_names
        except Exception as e:
            print(f"Error loading encodings: {e}")
            return None, None

def main():
    trainer = FaceTrainer()
    trainer.train_face_encodings()

if __name__ == "__main__":
    main()