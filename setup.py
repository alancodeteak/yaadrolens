#!/usr/bin/env python3
"""
Setup script for Face Recognition Attendance System
"""
import os
import subprocess
import sys

def run_command(command, description):
    """Run a shell command with error handling"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def setup_environment():
    """Setup the development environment"""
    print("🚀 Setting up Face Recognition Attendance System")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Create virtual environment if it doesn't exist
    if not os.path.exists("venv"):
        if not run_command("python3 -m venv venv", "Creating virtual environment"):
            return False
    else:
        print("✅ Virtual environment already exists")
    
    # Activate virtual environment and install requirements
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate"
    else:  # Unix/Linux/macOS
        activate_cmd = "source venv/bin/activate"
    
    install_cmd = f"{activate_cmd} && pip install --upgrade pip && pip install -r requirements.txt"
    
    if not run_command(install_cmd, "Installing Python dependencies"):
        print("⚠️  Some packages might have failed to install")
        print("💡 Try running manually: pip install -r requirements.txt")
    
    # Create necessary directories
    directories = ["dataset", "uploads/employee_images", "uploads/attendance_images", "migrations"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Create .env file if it doesn't exist
    if not os.path.exists(".env"):
        env_content = """# Database Configuration
DATABASE_URL=postgresql://username:password@localhost/face_attendance

# Security
SECRET_KEY=your-secret-key-change-this-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Face Recognition Settings
RECOGNITION_THRESHOLD=0.68
DATASET_PATH=dataset
"""
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ Created .env file (please update with your settings)")
    
    # Setup sample dataset structure
    setup_sample_dataset()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📝 Next steps:")
    print("   1. Update .env with your database credentials")
    print("   2. Run database migrations: alembic upgrade head")
    print("   3. Add photos to dataset/ folders")
    print("   4. Test with: python test_deepface.py")
    print("   5. Run simple system: python simple_attendance.py")
    print("   6. Run full API: python run.py")
    
    return True

def setup_sample_dataset():
    """Setup sample dataset structure"""
    sample_persons = ["john_doe", "jane_smith", "alice_johnson", "bob_wilson"]
    
    for person in sample_persons:
        person_dir = os.path.join("dataset", person)
        os.makedirs(person_dir, exist_ok=True)
        
        # Create README for each person
        readme_path = os.path.join(person_dir, "README.txt")
        if not os.path.exists(readme_path):
            with open(readme_path, "w") as f:
                f.write(f"Photos for {person.replace('_', ' ').title()}\n")
                f.write("=" * 40 + "\n\n")
                f.write("Add 5-10 photos of this person here:\n")
                f.write("• Supported formats: .jpg, .jpeg, .png\n")
                f.write("• Different angles and expressions\n")
                f.write("• Good lighting and clear face\n")
                f.write("• Examples: 1.jpg, 2.jpg, 3.jpg, etc.\n\n")
                f.write("Tips for best accuracy:\n")
                f.write("• Face should be clearly visible\n")
                f.write("• Avoid heavy shadows or glare\n")
                f.write("• Include photos with/without glasses\n")
                f.write("• Various facial expressions\n")
    
    print(f"✅ Created sample dataset structure with {len(sample_persons)} persons")

if __name__ == "__main__":
    if setup_environment():
        sys.exit(0)
    else:
        sys.exit(1)
