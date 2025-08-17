#!/usr/bin/env python3
"""
Dual-Person Smart Attendance & Gaze-Based Attentiveness Monitoring System
Main entry point for the application
"""
import mysql.connector
print("Imported mysql.connector successfully")
try:
    test_conn = mysql.connector.connect(
        host='localhost',
        #enter ur user id and password
        user='',
        password='',
        database='dualeye',
        connection_timeout=5
    )
    print("Test connection in mainc.py successful")
    test_conn.close()
except Exception as e:
    print("Test connection in mainc.py failed:", e)

import sys
import time
import logging
import cv2
import pandas as pd
import numpy as np
import face_recognition
from datetime import datetime
from pathlib import Path

from config import *
from database.db_manager import DatabaseManager
from gui.gui_components import show_popup, show_error, select_people, show_attentiveness_report
from gaze_tracking import GazeTracking
from face_recognition_module.train import FaceTrainer
from utils.file_utils import download_shape_predictor_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DualPersonGazeTracker:

    def analyze_attentiveness(self, student_name):
        """Phase 6: Analyze attentiveness from gaze data"""
        conn = self.db_manager.create_database_connection()
        if not conn:
            return None, "Database connection failed"

        try:
            # Get gaze data from database
            query = """
                SELECT timestamp, left_pupil, right_pupil 
                FROM gaze_tracking 
                WHERE name = %s 
                ORDER BY timestamp
            """
            df = pd.read_sql(query, conn, params=[student_name])

            if df.empty:
                return None, "No data available"

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

            total_duration = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
            if total_duration <= 0:
                return None, "Invalid time duration"

            # Calculate non-attentive periods
            non_attentive_duration = 0.0
            in_non_attentive_period = False
            period_start = None

            for i in range(len(df)):
                current_time = df['timestamp'].iloc[i]
                left_pupil = df['left_pupil'].iloc[i]
                right_pupil = df['right_pupil'].iloc[i]

                # Check if both pupils are None (not detected)
                if left_pupil == 'None' and right_pupil == 'None':
                    if not in_non_attentive_period:
                        period_start = current_time
                        in_non_attentive_period = True
                else:
                    if in_non_attentive_period:
                        period_end = current_time
                        period_duration = (period_end - period_start).total_seconds()

                        # Only count periods longer than threshold
                        if period_duration > ATTENTION_THRESHOLD:
                            non_attentive_duration += period_duration

                        in_non_attentive_period = False

            # Handle case where session ends during non-attentive period
            if in_non_attentive_period:
                period_end = df['timestamp'].iloc[-1]
                period_duration = (period_end - period_start).total_seconds()
                if period_duration > ATTENTION_THRESHOLD:
                    non_attentive_duration += period_duration

            # Calculate attentiveness percentage
            attentive_duration = total_duration - non_attentive_duration
            attentiveness_percent = (attentive_duration / total_duration) * 100

            # Determine productivity category
            if attentiveness_percent >= 70:
                productivity = "Highly Productive"
            elif attentiveness_percent >= 40:
                productivity = "Moderately Productive"
            else:
                productivity = "Not Productive"

            # Save to database
            current_date = datetime.now().date()
            self.db_manager.save_attentiveness_to_db(student_name, current_date, 
                                                   attentiveness_percent, productivity)

            return attentiveness_percent, productivity

        except Exception as e:
            logger.error(f"Error analyzing attentiveness: {e}")
            return None, str(e)
        finally:
            if conn.is_connected():
                conn.close()

    def dual_gaze_tracking(self, selected_people, duration=GAZE_TRACKING_DURATION):
        """Phase 5: Dual gaze tracking for selected people"""
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            show_error("Cannot open camera!")
            return pd.DataFrame()

        # Initialize gaze trackers
        trackers = {}
        for person in selected_people:
            try:
                trackers[person] = GazeTracking()
            except Exception as e:
                show_error(f"Failed to initialize gaze tracker for {person}: {e}")
                return pd.DataFrame()

        gaze_df = pd.DataFrame(columns=["Name", "Timestamp", "Left_Pupil", "Right_Pupil", "Gaze_Direction"])

        show_popup(f"Starting gaze tracking for {' and '.join(selected_people)} for {duration} seconds...")
        start_time = time.time()

        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                break

            # Face detection and recognition
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            # Match faces to selected people
            face_matches = {}
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(
                    self.known_encodings, face_encoding, tolerance=FACE_RECOGNITION_TOLERANCE
                )

                if True in matches:
                    face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                    best_match_index = face_distances.argmin()
                    if matches[best_match_index]:
                        name = self.known_names[best_match_index]
                        if name in selected_people:
                            face_matches[(top, right, bottom, left)] = name

            # Process gaze tracking for each matched face
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            for (top, right, bottom, left), person_name in face_matches.items():
                # Scale coordinates back to full frame
                y1, y2, x1, x2 = top*4, bottom*4, left*4, right*4

                # Extract face region with margin
                margin = 20
                y1_margin = max(0, y1 - margin)
                y2_margin = min(frame.shape[0], y2 + margin)
                x1_margin = max(0, x1 - margin)
                x2_margin = min(frame.shape[1], x2 + margin)

                face_crop = frame[y1_margin:y2_margin, x1_margin:x2_margin]

                if face_crop.size > 0:
                    # Update gaze tracker
                    trackers[person_name].refresh(face_crop)

                    # Get gaze information
                    gaze_direction = "None"
                    if trackers[person_name].is_blinking():
                        gaze_direction = "Blinking"
                    elif trackers[person_name].is_right():
                        gaze_direction = "Looking right"
                    elif trackers[person_name].is_left():
                        gaze_direction = "Looking left"
                    elif trackers[person_name].is_center():
                        gaze_direction = "Looking center"

                    # Get pupil coordinates
                    left_pupil = trackers[person_name].pupil_left_coords()
                    right_pupil = trackers[person_name].pupil_right_coords()

                    # Save data
                    new_row = pd.DataFrame({
                        "Name": [person_name],
                        "Timestamp": [current_time],
                        "Left_Pupil": [left_pupil],
                        "Right_Pupil": [right_pupil],
                        "Gaze_Direction": [gaze_direction],
                    })
                    gaze_df = pd.concat([gaze_df, new_row], ignore_index=True)

                    # Save to database
                    self.db_manager.save_gaze_to_db(person_name, current_time, left_pupil, right_pupil, gaze_direction)

                    # Draw visualization
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{person_name}: {gaze_direction}", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Display frame info
            cv2.putText(frame, f"Tracking: {' & '.join(selected_people)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            remaining = int(duration - (time.time() - start_time))
            cv2.putText(frame, f"Time remaining: {remaining}s", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Dual Gaze Tracking Phase", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Save gaze data
        try:
            gaze_df.to_excel(GAZE_DATA_PATH, index=False)
            logger.info(f"Gaze data saved to {GAZE_DATA_PATH}")
        except Exception as e:
            logger.error(f"Failed to save gaze data: {e}")

        show_popup(f"Gaze tracking completed for {' and '.join(selected_people)}!")
        return gaze_df
    def verify_people(self, selected_people):
        """Phase 4: Verify both selected people are present"""
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            show_error("Cannot open camera!")
            return False

        verified_people = set()
        max_attempts = MAX_VERIFICATION_ATTEMPTS
        attempt = 0

        while len(verified_people) < 2 and attempt < max_attempts:
            attempt += 1
            show_popup(f"Verification attempt {attempt}/{max_attempts}. "
                      f"Please ensure both {' and '.join(selected_people)} are visible.")

            start_time = time.time()
            temp_verified = set()

            while time.time() - start_time < VERIFICATION_DURATION:
                ret, frame = cap.read()
                if not ret:
                    break

                # Face detection and recognition
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matches = face_recognition.compare_faces(
                        self.known_encodings, face_encoding, tolerance=FACE_RECOGNITION_TOLERANCE
                    )
                    name = "Unknown"

                    if True in matches:
                        face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                        best_match_index = face_distances.argmin()
                        if matches[best_match_index]:
                            name = self.known_names[best_match_index]
                            if name in selected_people:
                                temp_verified.add(name)

                    # Draw rectangle and label
                    top *= 4; right *= 4; bottom *= 4; left *= 4
                    color = (0, 255, 0) if name in selected_people else (0, 0, 255)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(frame, name, (left, bottom + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Show verification status
                status_text = f"Verified: {', '.join(temp_verified)} | Need: {', '.join(set(selected_people) - temp_verified)}"
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                remaining = int(VERIFICATION_DURATION - (time.time() - start_time))
                cv2.putText(frame, f"Time: {remaining}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("Verification Phase", frame)

                if len(temp_verified) == 2:
                    verified_people.update(temp_verified)
                    show_popup(f"Both {' and '.join(selected_people)} verified successfully!")
                    break

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
        return len(verified_people) == 2
    def __init__(self):
        print("[DEBUG] Entered __init__.")
        self.db_manager = DatabaseManager()
        print("[DEBUG] DatabaseManager created.")
        self.face_trainer = FaceTrainer()
        print("[DEBUG] FaceTrainer created.")
        self.known_encodings = None
        self.known_names = None
        self.initialize_system()
        print("[DEBUG] Finished __init__.")


    def initialize_system(self):
        logger.info("Initializing Dual Person Gaze Tracking System...")

        try:
            print("[DEBUG] Checking model download...")
            if not download_shape_predictor_model():
                print("[DEBUG] Failed at download_shape_predictor_model.")
                show_error("Failed to download shape predictor model!")
                sys.exit(1)
            print("[DEBUG] Shape predictor model checked.")

            print("[DEBUG] Initializing database tables...")
            result = self.db_manager.initialize_database_tables()
            print(f"[DEBUG] Database initialization result: {result}")
            if not result:
                print("[DEBUG] Failed at database initialization.")
                show_error("Failed to initialize database!")
                sys.exit(1)
            print("[DEBUG] Database initialized.")

            print("[DEBUG] Loading face encodings...")
            self.known_encodings, self.known_names = self.face_trainer.load_encodings()
            if self.known_encodings is None:
                print("[DEBUG] No face encodings found! Please run training.")
                show_error("No face encodings found! Please run face detection and training first.")
                sys.exit(1)
            print("[DEBUG] Face encodings loaded.")
            logger.info(f"Loaded {len(self.known_encodings)} face encodings")
            print("[DEBUG] Finished initialize_system.")
        except Exception as e:
            print("[DEBUG] Exception in initialize_system:", str(e))
            import traceback
            traceback.print_exc()

    
    def capture_attendance(self, duration=ATTENDANCE_DURATION):
        print("[DEBUG] Starting attendance capture.")
        attendance_df = pd.DataFrame(columns=["Name", "Status", "Date", "Time"])
        recognized_names = set()

        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            show_error("Cannot open camera!")
            return attendance_df

        show_popup(f"Attendance phase started. Detecting faces for {duration} seconds...")
        start_time = time.time()

        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture frame")
                break

            if frame.dtype != np.uint8:
                frame = cv2.convertScaleAbs(frame)

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(
                    self.known_encodings, face_encoding, tolerance=FACE_RECOGNITION_TOLERANCE
                )
                name = "Unknown"

                if True in matches:
                    face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                    best_match_index = face_distances.argmin()
                    if matches[best_match_index]:
                        name = self.known_names[best_match_index]

                        if name not in recognized_names:
                            recognized_names.add(name)
                            current_time = datetime.now()
                            date = current_time.strftime("%Y-%m-%d")
                            time_str = current_time.strftime("%H:%M:%S")

                            new_row = pd.DataFrame({
                                "Name": [name],
                                "Status": ["Present"],
                                "Date": [date],
                                "Time": [time_str]
                            })
                            attendance_df = pd.concat([attendance_df, new_row], ignore_index=True)

                            self.db_manager.save_attendance_to_db(name, "Present", date, time_str)
                            show_popup(f"{name} marked present!")

                # Draw rectangle and name
                top *= 4; right *= 4; bottom *= 4; left *= 4
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, bottom + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Show remaining time
            remaining = int(duration - (time.time() - start_time))
            cv2.putText(frame, f"Time remaining: {remaining}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Attendance Phase", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        try:
            attendance_df.to_excel(ATTENDANCE_PATH, index=False)
            logger.info(f"Attendance data saved to {ATTENDANCE_PATH}")
        except Exception as e:
            logger.error(f"Failed to save attendance: {e}")

        show_popup("Attendance phase completed!")
        print("[DEBUG] Attendance phase completed.")
        return attendance_df

    def run(self):
        print("[DEBUG] Entered DualPersonGazeTracker.run()")
        logger.info("Starting Dual Person Gaze Tracking System")
        try:
            input("Press Enter to start the program...")
            attendance_df = self.capture_attendance()
            if attendance_df.empty:
                print("[DEBUG] No attendees found in attendance df.")
                show_error("No attendees found. Exiting program.")
                return
            print("[DEBUG] attendance_df before select_people:")
            print(attendance_df)
            selected_people = select_people(attendance_df)
            print(f"[DEBUG] Selected_people: {selected_people}")
            if len(selected_people) != 2:
                show_error("Two people must be selected. Exiting program.")
                return
            logger.info(f"Selected people for gaze tracking: {selected_people}")
            if not self.verify_people(selected_people):
                print("[DEBUG] Verification of selected people failed or cancelled.")
                show_error("Could not verify both selected people. Exiting program.")
                return
            gaze_df = self.dual_gaze_tracking(selected_people)
            print(f"[DEBUG] Finished dual gaze tracking. Gaze df shape: {gaze_df.shape}")
            for person_name in selected_people:
                show_popup(f"Analyzing attentiveness for {person_name}...")
                percent, productivity = self.analyze_attentiveness(person_name)
                print(f"[DEBUG] Attentiveness for {person_name}: {percent}, {productivity}")
                if percent is not None:
                    show_attentiveness_report(person_name, percent, productivity)
                else:
                    show_error(f"Could not analyze attentiveness for {person_name}: {productivity}")
            show_popup("Program completed successfully!")
            print("[DEBUG] Program completed successfully.")
        except KeyboardInterrupt:
            logger.info("Program interrupted by user")
            print("[DEBUG] KeyboardInterrupt caught.")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"[DEBUG] Exception in run(): {e}")
            show_error(f"An error occurred: {e}")

def main():
    print("[DEBUG] main() function entered.")
    try:
        tracker = DualPersonGazeTracker()
        print("[DEBUG] DualPersonGazeTracker instantiated.")
        tracker.run()
        print("[DEBUG] tracker.run() finished.")
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        print(f"[DEBUG] Exception in main(): {e}")
        show_error(f"System initialization failed: {e}")

if __name__ == "__main__":
    print("[DEBUG] __main__ entry point triggered.")
    main()

