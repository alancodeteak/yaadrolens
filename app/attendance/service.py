import os
import uuid
import json
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, func
from fastapi import HTTPException, status, UploadFile
from app.attendance.models import AttendanceLog, AttendanceType
from app.employees.models import Employee, EmployeeEmbedding
from app.employees.service import EmployeeService
from app.face_recognition.deepface_service import deepface_service
from app.face_recognition.cached_recognition_service import CachedFaceRecognitionService
from app.core.redis_service import redis_service
from app.core.config import settings
from app.face_recognition.face_quality_utils import face_quality_validator
from app.face_recognition.smart_recognition_service import smart_recognition_service


class AttendanceService:
    def __init__(self, db: Session):
        self.db = db
        self.employee_service = EmployeeService(db)
        self.upload_dir = "uploads/attendance_images"
        os.makedirs(self.upload_dir, exist_ok=True)
        
        # Initialize cached recognition service if Redis is available
        if settings.enable_redis_cache and redis_service.is_available():
            self.cached_recognition = CachedFaceRecognitionService(redis_service, db)
            self.use_cache = True
        else:
            self.cached_recognition = None
            self.use_cache = False
    
    async def clock_in_out(self, image_file: UploadFile) -> Tuple[bool, str, Optional[str], Optional[str], Optional[AttendanceType], Optional[float], Optional[datetime]]:
        """
        Process clock in/out with face recognition.
        
        Returns:
            Tuple of (success, message, employee_id, employee_name, attendance_type, confidence_score, timestamp)
        """
        try:
            # Validate file type
            if not image_file.content_type or not image_file.content_type.startswith('image/'):
                return False, "Invalid image file", None, None, None, None, None
            
            # Save uploaded image
            file_extension = os.path.splitext(image_file.filename)[1]
            filename = f"attendance_{uuid.uuid4().hex}{file_extension}"
            file_path = os.path.join(self.upload_dir, filename)
            
            with open(file_path, "wb") as buffer:
                content = await image_file.read()
                buffer.write(content)
            
            # Step 1: Validate face quality before processing
            quality_result = face_quality_validator.validate_face_quality(file_path)
            
            if not quality_result.is_clear:
                # Clean up the saved file
                os.remove(file_path)
                issues_text = ", ".join(quality_result.issues)
                return False, f"Face quality check failed: {issues_text}. Please ensure clear face visibility.", None, None, None, None, None
            
            # Step 2: Face recognition (with caching if available)
            if self.use_cache and self.cached_recognition:
                # Use cached recognition service
                recognition_result = self.cached_recognition.recognize_face_cached(file_path)
                
                if recognition_result['success']:
                    employee_id = recognition_result['employee_id']
                    confidence = recognition_result['confidence']
                else:
                    # Clean up the saved file
                    os.remove(file_path)
                    return False, recognition_result.get('error', 'Face recognition failed'), None, None, None, None, None
            else:
                # Fallback to original method
                uploaded_embedding = deepface_service.extract_embedding(file_path)
                
                if uploaded_embedding is None:
                    # Clean up the saved file
                    os.remove(file_path)
                    return False, "Face not detected clearly in the uploaded image.", None, None, None, None, None
                
                # Step 3: Find best match in database embeddings
                recognition_result = self._find_best_embedding_match(uploaded_embedding)
                
                if recognition_result is None:
                    # Clean up the saved file
                    os.remove(file_path)
                    return False, "No matching employee found. Please ensure you are registered in the system.", None, None, None, None, None
                
                employee_id = recognition_result['employee_id']
                confidence = recognition_result['confidence']
            
            confidence_score = confidence
            
            # Get employee details
            employee = self.employee_service.get_employee(employee_id)
            if not employee or not employee.is_active:
                os.remove(file_path)
                return False, "Employee not found or inactive", None, None, None, confidence_score, None
            
            # Determine if this should be clock in or clock out
            attendance_type = self._determine_attendance_type(employee_id)
            
            # Create attendance record
            attendance_log = AttendanceLog(
                employee_id=employee_id,
                type=attendance_type,
                confidence_score=f"{confidence_score:.3f}",
                image_path=file_path
            )
            
            self.db.add(attendance_log)
            self.db.commit()
            self.db.refresh(attendance_log)
            
            return True, f"Successfully clocked {attendance_type.value}", employee_id, employee.name, attendance_type, confidence_score, attendance_log.timestamp
            
        except Exception as e:
            self.db.rollback()
            # Clean up file if it exists
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
            return False, f"Error processing attendance: {str(e)}", None, None, None, None, None
    
    def _determine_attendance_type(self, employee_id: str) -> AttendanceType:
        """Determine if this should be a clock in or clock out based on last attendance."""
        # Get the last attendance record for this employee today
        today = datetime.now().date()
        last_attendance = self.db.query(AttendanceLog).filter(
            and_(
                AttendanceLog.employee_id == employee_id,
                func.date(AttendanceLog.timestamp) == today
            )
        ).order_by(AttendanceLog.timestamp.desc()).first()
        
        # If no attendance today or last was OUT, this should be IN
        if not last_attendance or last_attendance.type == AttendanceType.OUT:
            return AttendanceType.IN
        else:
            return AttendanceType.OUT
    
    def _find_best_embedding_match(self, uploaded_embedding: np.ndarray) -> Optional[Dict]:
        """
        Find the best matching employee embedding in the database using cosine similarity.
        
        Args:
            uploaded_embedding: Face embedding from uploaded image
            
        Returns:
            Dict with employee_id and confidence, or None if no good match
        """
        try:
            # Get all employee embeddings from database
            all_embeddings = self.db.query(EmployeeEmbedding).join(Employee).filter(
                Employee.is_active == True
            ).all()
            
            if not all_embeddings:
                return None
            
            best_match = None
            best_similarity = 0.0
            similarity_threshold = 0.5  # Adjust this threshold as needed
            
            for db_embedding in all_embeddings:
                try:
                    # Parse the stored embedding (JSON string to numpy array)
                    stored_embedding = np.array(json.loads(db_embedding.embedding))
                    
                    # Calculate cosine similarity
                    similarity = self._cosine_similarity(uploaded_embedding, stored_embedding)
                    
                    if similarity > best_similarity and similarity > similarity_threshold:
                        best_similarity = similarity
                        best_match = {
                            'employee_id': str(db_embedding.employee_id),
                            'confidence': similarity,
                            'matched_embedding_id': db_embedding.id
                        }
                
                except (json.JSONDecodeError, ValueError) as e:
                    # Skip corrupted embeddings
                    continue
            
            return best_match
            
        except Exception as e:
            print(f"Error during face recognition: {str(e)}")
            return None
    
    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1, higher is more similar)
        """
        try:
            # Normalize the vectors
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            
            # Ensure result is in [0, 1] range
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            return 0.0
    
    def get_attendance_logs(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        employee_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[AttendanceLog]:
        """Get attendance logs for a date range."""
        query = self.db.query(AttendanceLog).join(Employee).filter(
            and_(
                AttendanceLog.timestamp >= start_date,
                AttendanceLog.timestamp <= end_date
            )
        )
        
        if employee_id:
            query = query.filter(AttendanceLog.employee_id == employee_id)
        
        return query.order_by(AttendanceLog.timestamp.desc()).offset(skip).limit(limit).all()
    
    def get_employee_attendance_report(
        self, 
        employee_id: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> dict:
        """Generate attendance report for a specific employee."""
        employee = self.employee_service.get_employee(employee_id)
        if not employee:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Employee not found"
            )
        
        # Get all attendance logs for the period
        attendance_logs = self.get_attendance_logs(start_date, end_date, employee_id)
        
        # Calculate total days and hours
        total_days = len(set(log.timestamp.date() for log in attendance_logs))
        total_hours = self._calculate_total_hours(attendance_logs)
        
        return {
            "employee_id": employee_id,
            "employee_name": employee.name,
            "employee_email": employee.email,
            "total_days": total_days,
            "total_hours": total_hours,
            "attendance_logs": attendance_logs
        }
    
    def _calculate_total_hours(self, attendance_logs: List[AttendanceLog]) -> float:
        """Calculate total hours worked from attendance logs."""
        # Group logs by date
        daily_logs = {}
        for log in attendance_logs:
            date = log.timestamp.date()
            if date not in daily_logs:
                daily_logs[date] = []
            daily_logs[date].append(log)
        
        total_hours = 0.0
        
        for date, logs in daily_logs.items():
            # Sort logs by timestamp
            logs.sort(key=lambda x: x.timestamp)
            
            # Calculate hours for this day
            clock_in_time = None
            for log in logs:
                if log.type == AttendanceType.IN and clock_in_time is None:
                    clock_in_time = log.timestamp
                elif log.type == AttendanceType.OUT and clock_in_time is not None:
                    # Calculate hours worked
                    hours_worked = (log.timestamp - clock_in_time).total_seconds() / 3600
                    total_hours += hours_worked
                    clock_in_time = None
        
        return round(total_hours, 2)
    
    def get_daily_attendance_summary(self, date: datetime) -> List[dict]:
        """Get daily attendance summary for all employees."""
        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = date.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        # Get all attendance logs for the day
        attendance_logs = self.get_attendance_logs(start_of_day, end_of_day)
        
        # Group by employee
        employee_summary = {}
        for log in attendance_logs:
            employee_id = str(log.employee_id)
            if employee_id not in employee_summary:
                employee_summary[employee_id] = {
                    "employee_id": employee_id,
                    "employee_name": log.employee.name,
                    "employee_email": log.employee.email,
                    "clock_in": None,
                    "clock_out": None,
                    "hours_worked": 0.0
                }
            
            if log.type == AttendanceType.IN:
                employee_summary[employee_id]["clock_in"] = log.timestamp
            elif log.type == AttendanceType.OUT:
                employee_summary[employee_id]["clock_out"] = log.timestamp
        
        # Calculate hours worked for each employee
        for employee_id, summary in employee_summary.items():
            if summary["clock_in"] and summary["clock_out"]:
                hours = (summary["clock_out"] - summary["clock_in"]).total_seconds() / 3600
                summary["hours_worked"] = round(hours, 2)
        
        return list(employee_summary.values())
    
    async def smart_clock_in_out(self, image_file: UploadFile) -> Dict[str, Any]:
        """
        Smart clock in/out with enhanced recognition features.
        
        Returns:
            Dict with detailed recognition results and recommendations
        """
        try:
            # Validate file type
            if not image_file.content_type or not image_file.content_type.startswith('image/'):
                return {
                    "success": False,
                    "error": "Invalid image file",
                    "retry_recommended": False,
                    "suggested_actions": ["Please upload a valid image file"]
                }
            
            # Save image
            file_extension = os.path.splitext(image_file.filename)[1]
            filename = f"attendance_{uuid.uuid4().hex}{file_extension}"
            file_path = os.path.join(self.upload_dir, filename)
            
            with open(file_path, "wb") as buffer:
                content = await image_file.read()
                buffer.write(content)
            
            # Get all employee embeddings
            employee_embeddings = self.db.query(EmployeeEmbedding).all()
            embeddings_data = [
                {
                    "employee_id": str(emb.employee_id),
                    "embedding": emb.embedding,
                    "image_path": emb.image_path
                }
                for emb in employee_embeddings
            ]
            
            if not embeddings_data:
                return {
                    "success": False,
                    "error": "No employees registered in the system",
                    "retry_recommended": False,
                    "suggested_actions": ["Contact HR to register employees"]
                }
            
            # Use smart recognition
            recognition_result = smart_recognition_service.recognize_face_smart(file_path, embeddings_data)
            
            if not recognition_result["success"]:
                # Clean up failed image
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                return {
                    "success": False,
                    "error": recognition_result["error"],
                    "confidence": recognition_result.get("confidence", 0.0),
                    "quality_score": recognition_result.get("quality_score", 0.0),
                    "retry_recommended": recognition_result.get("retry_recommended", True),
                    "suggested_actions": recognition_result.get("suggested_actions", []),
                    "metadata": recognition_result.get("metadata", {})
                }
            
            # Get employee details
            employee = self.db.query(Employee).filter(
                Employee.id == recognition_result["employee_id"]
            ).first()
            
            if not employee:
                return {
                    "success": False,
                    "error": "Employee not found",
                    "retry_recommended": True,
                    "suggested_actions": ["Contact HR to verify employee registration"]
                }
            
            # Determine attendance type
            last_attendance = self.db.query(AttendanceLog).filter(
                AttendanceLog.employee_id == employee.id
            ).order_by(AttendanceLog.timestamp.desc()).first()
            
            if last_attendance and last_attendance.type == AttendanceType.IN:
                attendance_type = AttendanceType.OUT
            else:
                attendance_type = AttendanceType.IN
            
            # Create attendance log
            attendance_log = AttendanceLog(
                employee_id=employee.id,
                type=attendance_type,
                confidence_score=str(recognition_result["confidence"]),
                image_path=file_path
            )
            
            self.db.add(attendance_log)
            self.db.commit()
            
            return {
                "success": True,
                "employee_id": str(employee.id),
                "employee_name": employee.name,
                "attendance_type": attendance_type.value,
                "confidence": recognition_result["confidence"],
                "quality_score": recognition_result.get("quality_score", 0.0),
                "retry_recommended": recognition_result.get("retry_recommended", False),
                "timestamp": attendance_log.timestamp.isoformat(),
                "metadata": recognition_result.get("metadata", {})
            }
            
        except Exception as e:
            self.db.rollback()
            # Clean up file on error
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
            
            return {
                "success": False,
                "error": f"System error: {str(e)}",
                "retry_recommended": True,
                "suggested_actions": ["Please try again or contact support"]
            }
