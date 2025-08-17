import mysql.connector
from mysql.connector import Error
from config import DB_CONFIG
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.config = DB_CONFIG
        
    def create_database_connection(self):
        """Create and return a database connection"""
        print("[DEBUG][DB] Attempting to connect to MySQL with config:", self.config)
        try:
            config_with_timeout = dict(self.config)
            config_with_timeout['connection_timeout'] = 5
            print("[DEBUG][DB] Before mysql.connector.connect() call")
            conn = mysql.connector.connect(**config_with_timeout)
            print("[DEBUG][DB] After mysql.connector.connect() call")
            print("[DEBUG][DB] MySQL connection established.")
            logger.info("Successfully connected to MySQL database")
            return conn
        except mysql.connector.Error as err:
            print(f"[DEBUG][DB] Error connecting to MySQL: {err}")
            logger.error(f"Error connecting to MySQL: {err}")
            return None
    
    def initialize_database_tables(self):
        print("[DEBUG][DB] Entering initialize_database_tables")
        try:
            print("[DEBUG][DB] Calling create_database_connection()...")
            conn = self.create_database_connection()
            print(f"[DEBUG][DB] Got connection: {conn}")
            if conn:
                try:
                    print("[DEBUG][DB] Creating cursor...")
                    cursor = conn.cursor()
                    print("[DEBUG][DB] Cursor created.")
                    print("[DEBUG][DB] Creating attendance table...")
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS attendance (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            name VARCHAR(100) NOT NULL,
                            status VARCHAR(20) NOT NULL,
                            date DATE NOT NULL,
                            time TIME NOT NULL
                        ) ENGINE=InnoDB;
                    ''')
                    print("[DEBUG][DB] Attendance table created (or already exists).")
                    print("[DEBUG][DB] Creating gaze_tracking table...")
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS gaze_tracking (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            name VARCHAR(100) NOT NULL,
                            timestamp DATETIME NOT NULL,
                            left_pupil VARCHAR(50),
                            right_pupil VARCHAR(50),
                            gaze_direction VARCHAR(50)
                        ) ENGINE=InnoDB;
                    ''')
                    print("[DEBUG][DB] Gaze_tracking table created (or already exists).")
                    print("[DEBUG][DB] Creating attentiveness table...")
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS attentiveness (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            name VARCHAR(100) NOT NULL,
                            date DATE NOT NULL,
                            attentiveness_percent FLOAT,
                            productivity VARCHAR(50)
                        ) ENGINE=InnoDB;
                    ''')
                    print("[DEBUG][DB] Attentiveness table created (or already exists).")
                    print("[DEBUG][DB] Committing changes...")
                    conn.commit()
                    print("[DEBUG][DB] Closing cursor...")
                    cursor.close()
                    print("[DEBUG][DB] Closing connection...")
                    conn.close()
                    print("[DEBUG][DB] Closed connection/cursor.")
                    return True
                except Exception as inner_err:
                    print(f"[DEBUG][DB] Exception during table creation: {inner_err}")
                    import traceback
                    traceback.print_exc()
                    return False
            else:
                print("[DEBUG][DB] No database connection.")
                return False
        except Exception as db_err:
            print(f"[DEBUG][DB] Exception at DB connection level: {db_err}")
            import traceback
            traceback.print_exc()
            return False

    
    
    def save_attendance_to_db(self, name, status, date, time_str):
        """Save attendance data to database"""
        conn = self.create_database_connection()
        if conn:
            try:
                cursor = conn.cursor()
                query = """
                    INSERT INTO attendance (name, status, date, time)
                    VALUES (%s, %s, %s, %s)
                """
                cursor.execute(query, (name, status, date, time_str))
                conn.commit()
                cursor.close()
                conn.close()
                logger.info(f"Saved attendance for {name}")
                return True
            except mysql.connector.Error as err:
                logger.error(f"Error saving attendance: {err}")
                return False
        return False
    
    def save_gaze_to_db(self, name, timestamp, left_pupil, right_pupil, gaze_direction):
        """Save gaze tracking data to database"""
        conn = self.create_database_connection()
        if conn:
            try:
                cursor = conn.cursor()
                
                left_pupil_str = str(left_pupil) if left_pupil is not None else "None"
                right_pupil_str = str(right_pupil) if right_pupil is not None else "None"
                
                query = """
                    INSERT INTO gaze_tracking (name, timestamp, left_pupil, right_pupil, gaze_direction)
                    VALUES (%s, %s, %s, %s, %s)
                """
                
                cursor.execute(query, (name, timestamp, left_pupil_str, right_pupil_str, gaze_direction))
                conn.commit()
                cursor.close()
                conn.close()
                return True
            except mysql.connector.Error as err:
                logger.error(f"Error saving gaze data: {err}")
                return False
        return False
    
    def save_attentiveness_to_db(self, name, date, percent, productivity):
        """Save attentiveness analysis to database"""
        conn = self.create_database_connection()
        if conn:
            try:
                cursor = conn.cursor()
                query = """
                    INSERT INTO attentiveness (name, date, attentiveness_percent, productivity)
                    VALUES (%s, %s, %s, %s)
                """
                cursor.execute(query, (name, date, percent, productivity))
                conn.commit()
                cursor.close()
                conn.close()
                logger.info(f"Saved attentiveness data for {name}")
                return True
            except mysql.connector.Error as err:
                logger.error(f"Error saving attentiveness data: {err}")
                return False
        return False