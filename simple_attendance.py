"""
Simple Face Recognition Attendance System
Using DeepFace + ArcFace for high accuracy recognition
"""
import cv2
import os
from deepface import DeepFace
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_DIR = "dataset"
ATTENDANCE_FILE = "attendance.csv"

def recognize_face(frame):
    """Recognize face in the given frame"""
    try:
        result = DeepFace.find(
            img_path=frame,
            db_path=DATASET_DIR,
            model_name="ArcFace",
            detector_backend="mtcnn",
            enforce_detection=True,
            silent=True
        )
        
        if len(result) > 0 and len(result[0]) > 0:
            # Get the best match
            best_match = result[0].iloc[0]
            distance = best_match["ArcFace_cosine"]
            similarity = 1 - distance  # Convert distance to similarity
            
            # Check if similarity meets threshold
            if similarity >= 0.68:  # ArcFace threshold
                person_dir = best_match['identity']
                person_id = os.path.basename(os.path.dirname(person_dir))
                
                logger.info(f"Face recognized: {person_id} (confidence: {similarity:.3f})")
                return person_id, similarity
        
        logger.info("No face match found")
        return None, 0.0
        
    except Exception as e:
        logger.error(f"Recognition error: {e}")
        return None, 0.0

def mark_attendance(name, confidence):
    """Mark attendance in CSV file"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create attendance file if it doesn't exist
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w") as f:
            f.write("Name,Time,Confidence\n")
    
    # Check if already marked today
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        with open(ATTENDANCE_FILE, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip header
                if line.strip():
                    parts = line.strip().split(",")
                    if len(parts) >= 2 and parts[0] == name and today in parts[1]:
                        print(f"[INFO] {name} already marked attendance today")
                        return
    except:
        pass
    
    # Mark attendance
    with open(ATTENDANCE_FILE, "a") as f:
        f.write(f"{name},{now},{confidence:.3f}\n")
    
    print(f"[INFO] ✅ Attendance marked: {name} at {now} (confidence: {confidence:.3f})")

def run_attendance_system():
    """Run the main attendance system"""
    print("🎯 Face Recognition Attendance System")
    print("📹 Starting camera... Press 'q' to quit")
    
    # Check if dataset exists
    if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
        print("❌ No dataset found! Please register some people first.")
        print(f"📁 Create folders in '{DATASET_DIR}/' with person names and add their photos")
        return
    
    # List registered people
    registered = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    print(f"👥 Registered people: {', '.join(registered)}")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    # Recognition cooldown to avoid spam
    last_recognition = {}
    cooldown_seconds = 5
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read frame")
            break
        
        # Save temporary frame for processing
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, frame)
        
        # Try to recognize face
        name, confidence = recognize_face(temp_path)
        
        if name:
            # Check cooldown
            current_time = datetime.now()
            if name not in last_recognition or \
               (current_time - last_recognition[name]).seconds >= cooldown_seconds:
                
                mark_attendance(name, confidence)
                last_recognition[name] = current_time
            
            # Display recognition result
            cv2.putText(frame, f"{name} ({confidence:.2f})", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No match", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Show frame
        cv2.imshow("Face Recognition Attendance", frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Remove temporary file
    if os.path.exists("temp_frame.jpg"):
        os.remove("temp_frame.jpg")
    
    print("👋 Attendance system stopped")

def setup_sample_dataset():
    """Setup sample dataset structure"""
    print("📁 Setting up sample dataset structure...")
    
    sample_persons = ["john_doe", "jane_smith", "alice_johnson"]
    
    for person in sample_persons:
        person_dir = os.path.join(DATASET_DIR, person)
        os.makedirs(person_dir, exist_ok=True)
        
        # Create a placeholder file
        placeholder_path = os.path.join(person_dir, "README.txt")
        with open(placeholder_path, "w") as f:
            f.write(f"Add photos of {person.replace('_', ' ').title()} here\n")
            f.write("Supported formats: .jpg, .jpeg, .png\n")
            f.write("Recommended: 5-10 photos with different angles and expressions\n")
    
    print(f"✅ Created sample dataset structure in '{DATASET_DIR}/'")
    print("📸 Add photos to each person's folder and run the system again")

if __name__ == "__main__":
    print("🚀 DeepFace + ArcFace Attendance System")
    print("=" * 50)
    
    # Check if dataset exists
    if not os.path.exists(DATASET_DIR) or not any(
        os.path.isdir(os.path.join(DATASET_DIR, d)) for d in os.listdir(DATASET_DIR)
    ):
        setup_sample_dataset()
    else:
        run_attendance_system()

