"""
Face Quality Validation Utils
Detects face quality, obstructions, and ensures clear face visibility
"""
import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
import os

class FaceQualityResult:
    """Result of face quality analysis"""
    def __init__(self, is_clear: bool, confidence: float, issues: List[str],
                 face_detected: bool, hands_detected: bool, brightness_ok: bool,
                 blur_ok: bool, face_area_ratio: float):
        self.is_clear = is_clear
        self.confidence = confidence
        self.issues = issues
        self.face_detected = face_detected
        self.hands_detected = hands_detected
        self.brightness_ok = brightness_ok
        self.blur_ok = blur_ok
        self.face_area_ratio = face_area_ratio
    
class FaceQualityValidator:
    """Validates face quality and detects obstructions using OpenCV"""
    
    def __init__(self, strict_mode=False):
        # Initialize OpenCV cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Strict mode for enhanced security (hand detection)
        self.strict_mode = strict_mode
        
        # Try to load hand cascade (if available)
        self.hand_cascade = None
        try:
            # Some OpenCV installations might have hand cascade
            hand_cascade_path = cv2.data.haarcascades + 'haarcascade_hand.xml'
            if os.path.exists(hand_cascade_path):
                self.hand_cascade = cv2.CascadeClassifier(hand_cascade_path)
        except:
            pass
        
        # Quality thresholds - Made more lenient
        self.min_brightness = 30
        self.max_brightness = 230
        self.min_face_area_ratio = 0.03  # Face should be at least 3% of image
        self.max_blur_threshold = 50.0  # Laplacian variance threshold (lower = more lenient)
        
    def validate_face_quality(self, image_path: str) -> FaceQualityResult:
        """
        Validate face quality from image file
        
        Args:
            image_path: Path to image file
            
        Returns:
            FaceQualityResult with validation details
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return FaceQualityResult(
                    is_clear=False, confidence=0.0, issues=["Cannot load image"],
                    face_detected=False, hands_detected=False, brightness_ok=False,
                    blur_ok=False, face_area_ratio=0.0
                )
            
            return self.validate_face_quality_from_array(image)
            
        except Exception as e:
            return FaceQualityResult(
                is_clear=False, confidence=0.0, issues=[f"Validation error: {str(e)}"],
                face_detected=False, hands_detected=False, brightness_ok=False,
                blur_ok=False, face_area_ratio=0.0
            )
    
    def validate_face_quality_from_array(self, image: np.ndarray) -> FaceQualityResult:
        """
        Validate face quality from image array
        
        Args:
            image: OpenCV image array (BGR)
            
        Returns:
            FaceQualityResult with validation details
        """
        issues = []
        
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 1. Face Detection
        face_detected, face_area_ratio, face_bbox = self._detect_face(rgb_image)
        if not face_detected:
            issues.append("No clear face detected")
        elif face_area_ratio < self.min_face_area_ratio:
            issues.append("Face too small in image")
        
        # 2. Hand/Obstruction Detection (only in strict mode)
        hands_detected, hand_near_face = False, False
        if self.strict_mode:
            hands_detected, hand_near_face = self._detect_hands_near_face(rgb_image, face_bbox)
            if hands_detected and hand_near_face:
                issues.append("Hands or fingers detected near face")
        
        # 3. Brightness Check
        brightness_ok, avg_brightness = self._check_brightness(image)
        if not brightness_ok:
            if avg_brightness < self.min_brightness:
                issues.append("Image too dark")
            else:
                issues.append("Image too bright")
        
        # 4. Blur Detection
        blur_ok, blur_score = self._check_blur(image)
        if not blur_ok:
            issues.append("Image too blurry")
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(
            face_detected, face_area_ratio, hands_detected and hand_near_face,
            brightness_ok, blur_ok
        )
        
        # Determine if face is clear - MUCH more lenient approach
        # Only reject if there are major issues
        is_clear = (
            face_detected and 
            face_area_ratio >= self.min_face_area_ratio
            # Removed hand detection and other checks - too many false positives
            # Only require basic face detection and size
        )
        
        return FaceQualityResult(
            is_clear=is_clear,
            confidence=confidence,
            issues=issues,
            face_detected=face_detected,
            hands_detected=hands_detected and hand_near_face,
            brightness_ok=brightness_ok,
            blur_ok=blur_ok,
            face_area_ratio=face_area_ratio
        )
    
    def _detect_face(self, rgb_image: np.ndarray) -> Tuple[bool, float, Optional[Dict]]:
        """Detect face and calculate area ratio using OpenCV"""
        try:
            # Convert RGB to grayscale for cascade detection
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
            )
            
            if len(faces) > 0:
                # Use the largest face
                face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = face
                
                # Calculate face area ratio
                image_area = rgb_image.shape[0] * rgb_image.shape[1]
                face_area = w * h
                face_area_ratio = face_area / image_area
                
                face_bbox = {
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h
                }
                
                return True, face_area_ratio, face_bbox
            
            return False, 0.0, None
            
        except Exception as e:
            print(f"Face detection error: {e}")
            return False, 0.0, None
    
    def _detect_hands_near_face(self, rgb_image: np.ndarray, face_bbox: Optional[Dict]) -> Tuple[bool, bool]:
        """Detect potential obstructions near face using advanced image analysis"""
        try:
            # Convert RGB to grayscale
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            
            # If no face detected, can't determine obstructions
            if not face_bbox:
                return False, False
            
            # Extract face region
            x, y, w, h = face_bbox['x'], face_bbox['y'], face_bbox['width'], face_bbox['height']
            
            # Expand region to check for hands/obstructions around face
            padding = int(min(w, h) * 0.3)  # 30% padding around face
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(gray.shape[1], x + w + padding)
            y_end = min(gray.shape[0], y + h + padding)
            
            face_region = gray[y_start:y_end, x_start:x_end]
            face_only = gray[y:y+h, x:x+w]
            
            # Method 1: Eye detection - if eyes are not visible, likely obstructed
            eyes_detected = self._detect_eyes_in_face(face_only)
            
            # Method 2: Edge density analysis - hands/fingers create more edges
            edge_density_suspicious = self._analyze_edge_density(face_region, face_only)
            
            # Method 3: Skin color analysis (simplified)
            skin_color_suspicious = self._analyze_skin_color_variance(rgb_image, face_bbox)
            
            # More conservative detection - require multiple indicators
            # Only flag as hands detected if eyes are NOT detected AND at least one other indicator
            hands_detected = (not eyes_detected) and (edge_density_suspicious or skin_color_suspicious)
            hand_near_face = hands_detected
            
            return hands_detected, hand_near_face
            
        except Exception as e:
            print(f"Obstruction detection error: {e}")
            return False, False
    
    def _detect_eyes_in_face(self, face_gray: np.ndarray) -> bool:
        """Detect if eyes are visible in the face region"""
        try:
            # More lenient eye detection parameters
            eyes = self.eye_cascade.detectMultiScale(
                face_gray, scaleFactor=1.05, minNeighbors=2, minSize=(5, 5)
            )
            # If no eyes detected with strict params, try more lenient
            if len(eyes) == 0:
                eyes = self.eye_cascade.detectMultiScale(
                    face_gray, scaleFactor=1.3, minNeighbors=1, minSize=(3, 3)
                )
            # Should detect at least one eye for clear face
            return len(eyes) >= 1
        except:
            # If eye detection fails, assume eyes are present (be lenient)
            return True
    
    def _analyze_edge_density(self, face_region: np.ndarray, face_only: np.ndarray) -> bool:
        """Analyze edge density to detect potential hand obstructions"""
        try:
            # Calculate edge density in expanded region vs face only
            edges_region = cv2.Canny(face_region, 50, 150)
            edges_face = cv2.Canny(face_only, 50, 150)
            
            region_density = np.sum(edges_region) / edges_region.size
            face_density = np.sum(edges_face) / edges_face.size
            
            # If region has significantly more edges than face, likely obstruction
            density_ratio = region_density / (face_density + 1e-6)  # Avoid division by zero
            
            return density_ratio > 3.0  # Much higher threshold - less sensitive
        except:
            return False
    
    def _analyze_skin_color_variance(self, rgb_image: np.ndarray, face_bbox: Dict) -> bool:
        """Analyze skin color variance to detect potential hand presence"""
        try:
            x, y, w, h = face_bbox['x'], face_bbox['y'], face_bbox['width'], face_bbox['height']
            
            # Extract face region
            face_region = rgb_image[y:y+h, x:x+w]
            
            # Convert to HSV for better skin detection
            hsv_face = cv2.cvtColor(face_region, cv2.COLOR_RGB2HSV)
            
            # Calculate color variance in the face region
            h_var = np.var(hsv_face[:, :, 0])  # Hue variance
            s_var = np.var(hsv_face[:, :, 1])  # Saturation variance
            
            # High variance might indicate multiple skin tones (hand + face)
            # This is a simplified heuristic
            total_variance = h_var + s_var
            
            return total_variance > 1500  # Much higher threshold - less sensitive
        except:
            return False
    
    def _check_brightness(self, image: np.ndarray) -> Tuple[bool, float]:
        """Check if image brightness is adequate"""
        try:
            # Convert to grayscale and calculate average brightness
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            
            brightness_ok = self.min_brightness <= avg_brightness <= self.max_brightness
            
            return brightness_ok, avg_brightness
            
        except Exception as e:
            print(f"Brightness check error: {e}")
            return False, 0.0
    
    def _check_blur(self, image: np.ndarray) -> Tuple[bool, float]:
        """Check if image is too blurry using Laplacian variance"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance (measure of blur)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            blur_ok = laplacian_var > self.max_blur_threshold
            
            return blur_ok, laplacian_var
            
        except Exception as e:
            print(f"Blur check error: {e}")
            return False, 0.0
    
    def _calculate_confidence(self, face_detected: bool, face_area_ratio: float, 
                            hands_near_face: bool, brightness_ok: bool, blur_ok: bool) -> float:
        """Calculate overall confidence score"""
        score = 0.0
        
        # Face detection (40% weight)
        if face_detected:
            score += 0.4 * min(1.0, face_area_ratio / self.min_face_area_ratio)
        
        # No hands near face (20% weight)
        if not hands_near_face:
            score += 0.2
        
        # Brightness (20% weight)
        if brightness_ok:
            score += 0.2
        
        # Blur (20% weight)
        if blur_ok:
            score += 0.2
        
        return min(1.0, score)

# Global instance - Default: lenient mode (no hand detection)
face_quality_validator = FaceQualityValidator(strict_mode=False)

# Strict instance for high security (with hand detection)
face_quality_validator_strict = FaceQualityValidator(strict_mode=True)
