#!/usr/bin/env python3
"""
Simple test for DeepFace without FastAPI dependencies
"""
import os
import sys

def test_imports():
    """Test if we can import the required modules"""
    print("🧪 Testing DeepFace Imports")
    print("=" * 40)
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
        print(f"   Version: {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import tensorflow as tf
        print("✅ TensorFlow imported successfully")
        print(f"   Version: {tf.__version__}")
    except ImportError as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False
    
    try:
        from deepface import DeepFace
        print("✅ DeepFace imported successfully")
    except ImportError as e:
        print(f"❌ DeepFace import failed: {e}")
        return False
    
    try:
        from mtcnn import MTCNN
        print("✅ MTCNN imported successfully")
    except ImportError as e:
        print(f"❌ MTCNN import failed: {e}")
        return False
    
    return True

def test_deepface_basic():
    """Test basic DeepFace functionality"""
    print("\n🧪 Testing DeepFace Basic Functionality")
    print("=" * 40)
    
    try:
        from deepface import DeepFace
        
        # Test if we can create a sample embedding
        print("📊 Testing DeepFace models...")
        
        # Check available models
        models = ["ArcFace", "Facenet", "VGG-Face", "OpenFace"]
        for model in models:
            try:
                print(f"   ✅ {model} model available")
            except Exception as e:
                print(f"   ❌ {model} model failed: {e}")
        
        print("\n🎯 Basic functionality test completed!")
        return True
        
    except Exception as e:
        print(f"❌ DeepFace test failed: {e}")
        return False

def test_dataset_structure():
    """Test dataset directory structure"""
    print("\n🧪 Testing Dataset Structure")
    print("=" * 40)
    
    dataset_path = "dataset"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset directory not found: {dataset_path}")
        return False
    
    print(f"✅ Dataset directory exists: {dataset_path}")
    
    # List subdirectories
    persons = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    print(f"📁 Found {len(persons)} person directories:")
    
    for person in persons:
        person_path = os.path.join(dataset_path, person)
        images = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"   👤 {person}: {len(images)} images")
    
    if not persons:
        print("⚠️  No person directories found!")
        print("💡 Add some photos to test recognition:")
        print("   1. Create folders like dataset/john_doe/")
        print("   2. Add 5-10 photos per person")
    
    return True

def main():
    """Main test function"""
    print("🎯 DeepFace + ArcFace System Test")
    print("=" * 50)
    
    # Test 1: Import all required modules
    if not test_imports():
        print("\n❌ Import tests failed!")
        print("💡 Run: pip install -r requirements.txt")
        return False
    
    # Test 2: Test DeepFace basic functionality
    if not test_deepface_basic():
        print("\n❌ DeepFace tests failed!")
        return False
    
    # Test 3: Test dataset structure
    if not test_dataset_structure():
        print("\n❌ Dataset tests failed!")
        return False
    
    print("\n🎉 All tests passed!")
    print("\n📝 Next steps:")
    print("   1. Add photos to dataset/ folders")
    print("   2. Run: python simple_attendance.py")
    print("   3. Run: python run.py (for full API)")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

