#!/usr/bin/env python3
"""
Test script for DeepFace + ArcFace system
"""
import os
import sys
from app.face_recognition.deepface_service import deepface_service

def test_deepface_service():
    """Test the DeepFace service functionality"""
    print("🧪 Testing DeepFace + ArcFace Service")
    print("=" * 50)
    
    # Test 1: Check if service is initialized
    print("✅ DeepFace service initialized")
    print(f"📁 Dataset path: {deepface_service.dataset_path}")
    print(f"🎯 Model: {deepface_service.model_name}")
    print(f"🔍 Detector: {deepface_service.detector_backend}")
    print(f"📊 Threshold: {deepface_service.similarity_threshold}")
    
    # Test 2: Check registered persons
    persons = deepface_service.get_registered_persons()
    print(f"\n👥 Registered persons: {len(persons)}")
    for person in persons:
        print(f"   - {person}")
    
    if not persons:
        print("\n⚠️  No persons registered yet!")
        print("📝 To test recognition:")
        print("   1. Create folders in 'dataset/' with person names")
        print("   2. Add 5-10 photos per person")
        print("   3. Run this test again")
        return
    
    # Test 3: Try recognition with existing dataset
    print(f"\n🔍 Testing recognition...")
    
    # Look for test images in dataset
    test_images = []
    for person in persons[:2]:  # Test first 2 persons
        person_dir = os.path.join(deepface_service.dataset_path, person)
        images = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if images:
            test_image = os.path.join(person_dir, images[0])
            test_images.append((person, test_image))
    
    for expected_person, test_image in test_images:
        print(f"\n🎯 Testing recognition for: {expected_person}")
        result = deepface_service.recognize_face(test_image)
        
        if result:
            recognized_person = result['person_id']
            confidence = result['confidence']
            
            if recognized_person == expected_person:
                print(f"   ✅ Correctly recognized: {recognized_person} (confidence: {confidence:.3f})")
            else:
                print(f"   ❌ Incorrectly recognized: {recognized_person} (expected: {expected_person})")
        else:
            print(f"   ❌ No recognition result")
    
    print(f"\n🎉 DeepFace service test completed!")

if __name__ == "__main__":
    try:
        test_deepface_service()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure to install requirements: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

