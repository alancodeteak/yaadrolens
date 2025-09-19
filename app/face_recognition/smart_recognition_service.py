"""
Smart Face Recognition Service with Enhanced Features
- Dynamic threshold optimization
- Confidence scoring and retry mechanisms
- Quality validation
- Anti-spoofing detection
- Performance monitoring
"""

import os
import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import json
import logging
from datetime import datetime
from app.core.config import settings
from app.face_recognition.deepface_service import DeepFaceService
from app.face_recognition.face_quality_utils import face_quality_validator

logger = logging.getLogger(__name__)

class SmartRecognitionService:
    """
    Enhanced face recognition with smart features
    """
    
    def __init__(self, ml_metrics_service=None):
        self.deepface_service = DeepFaceService()
        self.ml_metrics_service = ml_metrics_service
        self.recognition_stats = {
            "total_attempts": 0,
            "successful_recognitions": 0,
            "failed_recognitions": 0,
            "retry_attempts": 0,
            "quality_rejections": 0,
            "spoofing_detections": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "manual_corrections": 0
        }
        
    def recognize_face_smart(self, image_path: str, employee_embeddings: List[Dict], attendance_log_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Smart face recognition with enhanced features
        
        Args:
            image_path: Path to the input image
            employee_embeddings: List of employee embeddings from database
            attendance_log_id: Optional attendance log ID for ground truth tracking
            
        Returns:
            Dict with recognition results and metadata
        """
        start_time = datetime.now()
        self.recognition_stats["total_attempts"] += 1
        
        try:
            # Step 1: Quality validation
            if settings.enable_quality_validation:
                quality_result = self._validate_image_quality(image_path)
                if not quality_result["is_valid"]:
                    self.recognition_stats["quality_rejections"] += 1
                    return {
                        "success": False,
                        "error": "Poor image quality",
                        "confidence": 0.0,
                        "quality_score": quality_result["score"],
                        "retry_recommended": True,
                        "metadata": {
                            "quality_issues": quality_result["issues"],
                            "timestamp": datetime.now().isoformat()
                        }
                    }
            
            # Step 2: Anti-spoofing detection
            if settings.enable_anti_spoofing:
                spoofing_result = self._detect_spoofing(image_path)
                if spoofing_result["is_spoofed"]:
                    self.recognition_stats["spoofing_detections"] += 1
                    return {
                        "success": False,
                        "error": "Potential spoofing detected",
                        "confidence": 0.0,
                        "spoofing_score": spoofing_result["score"],
                        "retry_recommended": False,
                        "metadata": {
                            "spoofing_indicators": spoofing_result["indicators"],
                            "timestamp": datetime.now().isoformat()
                        }
                    }
            
            # Step 3: Face recognition with confidence scoring
            recognition_result = self._recognize_with_confidence(image_path, employee_embeddings)
            
            # Step 4: Smart decision making
            if recognition_result["confidence"] >= settings.high_confidence_threshold:
                # High confidence - immediate success
                self.recognition_stats["successful_recognitions"] += 1
                return {
                    "success": True,
                    "employee_id": recognition_result["employee_id"],
                    "confidence": recognition_result["confidence"],
                    "quality_score": quality_result.get("score", 0.0),
                    "retry_recommended": False,
                    "metadata": {
                        "recognition_type": "high_confidence",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            elif recognition_result["confidence"] >= settings.recognition_threshold:
                # Medium confidence - success but suggest retry for better quality
                self.recognition_stats["successful_recognitions"] += 1
                return {
                    "success": True,
                    "employee_id": recognition_result["employee_id"],
                    "confidence": recognition_result["confidence"],
                    "quality_score": quality_result.get("score", 0.0),
                    "retry_recommended": recognition_result["confidence"] < settings.high_confidence_threshold,
                    "metadata": {
                        "recognition_type": "medium_confidence",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            elif recognition_result["confidence"] >= settings.low_confidence_threshold:
                # Low confidence - retry recommended
                self.recognition_stats["retry_attempts"] += 1
                return {
                    "success": False,
                    "error": "Low confidence recognition",
                    "confidence": recognition_result["confidence"],
                    "quality_score": quality_result.get("score", 0.0),
                    "retry_recommended": True,
                    "suggested_actions": [
                        "Try better lighting",
                        "Look directly at camera",
                        "Remove glasses if possible",
                        "Ensure face is clearly visible"
                    ],
                    "metadata": {
                        "recognition_type": "low_confidence",
                        "best_match": recognition_result.get("best_match"),
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            else:
                # Very low confidence - no match
                self.recognition_stats["failed_recognitions"] += 1
                return {
                    "success": False,
                    "error": "No matching face found",
                    "confidence": recognition_result["confidence"],
                    "quality_score": quality_result.get("score", 0.0),
                    "retry_recommended": True,
                    "suggested_actions": [
                        "Ensure you are registered in the system",
                        "Try different lighting conditions",
                        "Look directly at camera",
                        "Contact HR if issue persists"
                    ],
                    "metadata": {
                        "recognition_type": "no_match",
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
        except Exception as e:
            logger.error(f"Error in smart recognition: {str(e)}")
            return {
                "success": False,
                "error": f"Recognition error: {str(e)}",
                "confidence": 0.0,
                "retry_recommended": True,
                "metadata": {
                    "error_type": "system_error",
                    "timestamp": datetime.now().isoformat()
                }
            }
    
    def _validate_image_quality(self, image_path: str) -> Dict[str, Any]:
        """Validate image quality using existing face quality validator"""
        try:
            quality_result = face_quality_validator.validate_face_quality(image_path)
            return {
                "is_valid": quality_result.confidence > 0.6,
                "score": quality_result.confidence,
                "issues": quality_result.issues if hasattr(quality_result, 'issues') else []
            }
        except Exception as e:
            logger.warning(f"Quality validation failed: {str(e)}")
            return {
                "is_valid": True,  # Default to valid if validation fails
                "score": 0.5,
                "issues": ["quality_validation_failed"]
            }
    
    def _detect_spoofing(self, image_path: str) -> Dict[str, Any]:
        """Basic anti-spoofing detection"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {"is_spoofed": False, "score": 0.0, "indicators": []}
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Basic spoofing indicators
            indicators = []
            spoofing_score = 0.0
            
            # 1. Check for rectangular artifacts (photo of photo)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for rectangular contours that might indicate a photo
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Large enough contour
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 0.7 < aspect_ratio < 1.4:  # Roughly square/rectangular
                        indicators.append("rectangular_artifact")
                        spoofing_score += 0.3
            
            # 2. Check for uniform lighting (suspicious for photos)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_std = np.std(hist)
            if hist_std < 1000:  # Very uniform lighting
                indicators.append("uniform_lighting")
                spoofing_score += 0.2
            
            # 3. Check for lack of depth variation
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 100:  # Low texture variation
                indicators.append("low_texture_variation")
                spoofing_score += 0.2
            
            # 4. Check for screen reflection patterns
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            blue_pixels = cv2.countNonZero(blue_mask)
            total_pixels = image.shape[0] * image.shape[1]
            blue_ratio = blue_pixels / total_pixels
            
            if blue_ratio > 0.1:  # High blue content (screen reflection)
                indicators.append("screen_reflection")
                spoofing_score += 0.3
            
            is_spoofed = spoofing_score > 0.5
            
            return {
                "is_spoofed": is_spoofed,
                "score": min(spoofing_score, 1.0),
                "indicators": indicators
            }
            
        except Exception as e:
            logger.warning(f"Spoofing detection failed: {str(e)}")
            return {"is_spoofed": False, "score": 0.0, "indicators": []}
    
    def _recognize_with_confidence(self, image_path: str, employee_embeddings: List[Dict]) -> Dict[str, Any]:
        """Perform face recognition with confidence scoring"""
        try:
            # Get embedding for input image
            input_embedding = self.deepface_service.get_embedding(image_path)
            if input_embedding is None:
                return {"confidence": 0.0, "employee_id": None}
            
            best_match = None
            best_similarity = 0.0
            
            # Compare with all employee embeddings
            for emp_data in employee_embeddings:
                try:
                    stored_embedding = json.loads(emp_data["embedding"])
                    similarity = self._cosine_similarity(input_embedding, stored_embedding)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = {
                            "employee_id": emp_data["employee_id"],
                            "similarity": similarity,
                            "image_path": emp_data.get("image_path")
                        }
                except Exception as e:
                    logger.warning(f"Error processing embedding for employee {emp_data.get('employee_id')}: {str(e)}")
                    continue
            
            return {
                "confidence": best_similarity,
                "employee_id": best_match["employee_id"] if best_match else None,
                "best_match": best_match
            }
            
        except Exception as e:
            logger.error(f"Error in recognition with confidence: {str(e)}")
            return {"confidence": 0.0, "employee_id": None}
    
    def _cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            logger.warning(f"Error calculating cosine similarity: {str(e)}")
            return 0.0
    
    def get_recognition_stats(self) -> Dict[str, Any]:
        """Get recognition performance statistics"""
        total = self.recognition_stats["total_attempts"]
        if total == 0:
            return self.recognition_stats
        
        return {
            **self.recognition_stats,
            "success_rate": self.recognition_stats["successful_recognitions"] / total,
            "retry_rate": self.recognition_stats["retry_attempts"] / total,
            "quality_rejection_rate": self.recognition_stats["quality_rejections"] / total,
            "spoofing_detection_rate": self.recognition_stats["spoofing_detections"] / total
        }
    
    def reset_stats(self):
        """Reset recognition statistics"""
        self.recognition_stats = {
            "total_attempts": 0,
            "successful_recognitions": 0,
            "failed_recognitions": 0,
            "retry_attempts": 0,
            "quality_rejections": 0,
            "spoofing_detections": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "manual_corrections": 0
        }
    
    def record_manual_correction(self, was_false_positive: bool = False, was_false_negative: bool = False):
        """Record a manual correction for ML metrics tracking"""
        self.recognition_stats["manual_corrections"] += 1
        
        if was_false_positive:
            self.recognition_stats["false_positives"] += 1
        elif was_false_negative:
            self.recognition_stats["false_negatives"] += 1
        
        logger.info(f"Manual correction recorded: FP={was_false_positive}, FN={was_false_negative}")
    
    def get_enhanced_recognition_stats(self) -> Dict[str, Any]:
        """Get enhanced recognition performance statistics including ML metrics"""
        base_stats = self.get_recognition_stats()
        
        total = self.recognition_stats["total_attempts"]
        if total == 0:
            return base_stats
        
        # Add enhanced metrics
        base_stats.update({
            "false_positive_rate": self.recognition_stats["false_positives"] / total,
            "false_negative_rate": self.recognition_stats["false_negatives"] / total,
            "manual_correction_rate": self.recognition_stats["manual_corrections"] / total,
            "system_accuracy": (total - self.recognition_stats["false_positives"] - self.recognition_stats["false_negatives"]) / total
        })
        
        return base_stats

# Global instance
smart_recognition_service = SmartRecognitionService()
