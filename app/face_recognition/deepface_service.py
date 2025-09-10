"""
DeepFace + ArcFace Face Recognition Service
High-accuracy face recognition using state-of-the-art models
"""
import os
import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict
import pandas as pd
from deepface import DeepFace
import tempfile
import shutil
import logging

logger = logging.getLogger(__name__)

class DeepFaceService:
    """
    Face Recognition Service using DeepFace with ArcFace embeddings
    """
    
    def __init__(self, dataset_path: str = "dataset"):
        self.dataset_path = dataset_path
        self.model_name = "ArcFace"  # State-of-the-art face embeddings
        self.detector_backend = "mtcnn"  # Best face detection
        self.distance_metric = "cosine"
        self.similarity_threshold = 0.68  # ArcFace threshold (lower = more strict)
        
        # Ensure dataset directory exists
        os.makedirs(self.dataset_path, exist_ok=True)
        
        logger.info(f"DeepFace service initialized with {self.model_name} model")
    
    def register_person(self, person_id: str, images: List[str]) -> bool:
        """
        Register a person with multiple images
        
        Args:
            person_id: Unique identifier for the person
            images: List of image file paths
            
        Returns:
            bool: Success status
        """
        try:
            person_dir = os.path.join(self.dataset_path, person_id)
            os.makedirs(person_dir, exist_ok=True)
            
            # Copy images to person directory
            for i, image_path in enumerate(images):
                if os.path.exists(image_path):
                    dest_path = os.path.join(person_dir, f"{i+1}.jpg")
                    shutil.copy2(image_path, dest_path)
                    logger.info(f"Registered image {i+1} for person {person_id}")
            
            logger.info(f"Successfully registered person {person_id} with {len(images)} images")
            return True
            
        except Exception as e:
            logger.error(f"Error registering person {person_id}: {str(e)}")
            return False
    
    def recognize_face(self, image_path: str) -> Optional[Dict]:
        """
        Recognize a face in the given image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict with person_id and confidence, or None if no match
        """
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None
            
            if not os.listdir(self.dataset_path):
                logger.warning("No registered persons in dataset")
                return None
            
            # Use DeepFace.find for recognition (try lenient first for webcam images)
            try:
                results = DeepFace.find(
                    img_path=image_path,
                    db_path=self.dataset_path,
                    model_name=self.model_name,
                    detector_backend=self.detector_backend,
                    distance_metric=self.distance_metric,
                    enforce_detection=False,  # More lenient for webcam captures
                    silent=True
                )
            except Exception as lenient_error:
                logger.warning(f"Lenient detection failed, trying strict mode: {lenient_error}")
                results = DeepFace.find(
                    img_path=image_path,
                    db_path=self.dataset_path,
                    model_name=self.model_name,
                    detector_backend=self.detector_backend,
                    distance_metric=self.distance_metric,
                    enforce_detection=True,
                    silent=True
                )
            
            if len(results) > 0 and len(results[0]) > 0:
                # Get the best match
                best_match = results[0].iloc[0]
                distance = best_match[f"{self.model_name}_{self.distance_metric}"]
                
                # Convert distance to similarity (for cosine: similarity = 1 - distance)
                similarity = 1 - distance if self.distance_metric == "cosine" else distance
                
                if similarity >= self.similarity_threshold:
                    # Extract person ID from file path
                    identity_path = best_match['identity']
                    person_id = os.path.basename(os.path.dirname(identity_path))
                    
                    result = {
                        'person_id': person_id,
                        'confidence': float(similarity),
                        'distance': float(distance),
                        'matched_image': identity_path
                    }
                    
                    logger.info(f"Face recognized: {person_id} (confidence: {similarity:.3f})")
                    return result
                else:
                    logger.info(f"Face detected but confidence too low: {similarity:.3f}")
                    return None
            
            logger.info("No face match found in dataset")
            return None
            
        except Exception as e:
            logger.error(f"Error during face recognition: {str(e)}")
            return None
    
    def verify_face(self, person_id: str, image_path: str) -> Dict:
        """
        Verify if an image matches a specific person
        
        Args:
            person_id: ID of the person to verify against
            image_path: Path to the image to verify
            
        Returns:
            Dict with verification result and confidence
        """
        try:
            person_dir = os.path.join(self.dataset_path, person_id)
            
            if not os.path.exists(person_dir):
                return {'verified': False, 'confidence': 0.0, 'error': 'Person not found'}
            
            # Get all images for the person
            person_images = [
                os.path.join(person_dir, f) 
                for f in os.listdir(person_dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            
            if not person_images:
                return {'verified': False, 'confidence': 0.0, 'error': 'No images for person'}
            
            # Verify against the first image (DeepFace.verify works with single images)
            reference_image = person_images[0]
            
            result = DeepFace.verify(
                img1_path=image_path,
                img2_path=reference_image,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                distance_metric=self.distance_metric,
                enforce_detection=True
            )
            
            # Convert distance to similarity
            distance = result['distance']
            similarity = 1 - distance if self.distance_metric == "cosine" else distance
            verified = result['verified']
            
            logger.info(f"Face verification for {person_id}: {verified} (confidence: {similarity:.3f})")
            
            return {
                'verified': verified,
                'confidence': float(similarity),
                'distance': float(distance),
                'threshold': result['threshold']
            }
            
        except Exception as e:
            logger.error(f"Error during face verification: {str(e)}")
            return {'verified': False, 'confidence': 0.0, 'error': str(e)}
    
    def extract_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract face embedding from an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Face embedding as numpy array, or None if no face found
        """
        try:
            # Try with strict detection first
            try:
                embedding = DeepFace.represent(
                    img_path=image_path,
                    model_name=self.model_name,
                    detector_backend=self.detector_backend,
                    enforce_detection=True
                )
                
                if embedding and len(embedding) > 0:
                    return np.array(embedding[0]['embedding'])
                    
            except Exception as strict_error:
                logger.warning(f"Strict detection failed, trying lenient mode: {strict_error}")
                
                # Try with lenient detection
                embedding = DeepFace.represent(
                    img_path=image_path,
                    model_name=self.model_name,
                    detector_backend=self.detector_backend,
                    enforce_detection=False
                )
                
                if embedding and len(embedding) > 0:
                    logger.info("Face detected with lenient mode")
                    return np.array(embedding[0]['embedding'])
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting embedding: {str(e)}")
            return None
    
    def get_registered_persons(self) -> List[str]:
        """
        Get list of all registered person IDs
        
        Returns:
            List of person IDs
        """
        try:
            if not os.path.exists(self.dataset_path):
                return []
            
            persons = [
                d for d in os.listdir(self.dataset_path)
                if os.path.isdir(os.path.join(self.dataset_path, d))
            ]
            
            return persons
            
        except Exception as e:
            logger.error(f"Error getting registered persons: {str(e)}")
            return []
    
    def delete_person(self, person_id: str) -> bool:
        """
        Delete a person from the dataset
        
        Args:
            person_id: ID of the person to delete
            
        Returns:
            bool: Success status
        """
        try:
            person_dir = os.path.join(self.dataset_path, person_id)
            
            if os.path.exists(person_dir):
                shutil.rmtree(person_dir)
                logger.info(f"Deleted person {person_id} from dataset")
                return True
            
            logger.warning(f"Person {person_id} not found in dataset")
            return False
            
        except Exception as e:
            logger.error(f"Error deleting person {person_id}: {str(e)}")
            return False

# Global service instance
deepface_service = DeepFaceService()

