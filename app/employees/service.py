import os
import json
import uuid
import logging
import time
from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status, UploadFile
from app.employees.models import Employee, EmployeeEmbedding, TrainingPhoto
from app.employees.schemas import EmployeeCreate, EmployeeUpdate
from app.face_recognition.deepface_service import deepface_service
from app.face_recognition.face_quality_utils import face_quality_validator
from app.core.redis_service import redis_service

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmployeeService:
    def __init__(self, db: Session):
        self.db = db
        self.upload_dir = "uploads/employee_images"
        self.dataset_dir = "dataset"
        self.training_dir = "uploads/training_photos"
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.training_dir, exist_ok=True)
    
    def create_employee(self, employee_data: EmployeeCreate) -> Employee:
        """Create a new employee."""
        try:
            # Check if employee already exists
            existing_employee = self.db.query(Employee).filter(
                Employee.email == employee_data.email
            ).first()
            
            if existing_employee:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Employee with this email already exists"
                )
            
            # Create new employee
            db_employee = Employee(
                name=employee_data.name,
                email=employee_data.email,
                department=employee_data.department,
                position=employee_data.position,
                phone=employee_data.phone
            )
            
            self.db.add(db_employee)
            self.db.commit()
            self.db.refresh(db_employee)
            
            return db_employee
            
        except IntegrityError:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Employee with this email already exists"
            )
        except Exception as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error creating employee: {str(e)}"
            )
    
    async def add_employee_embeddings(self, employee_id, image_files: List[UploadFile]) -> List[EmployeeEmbedding]:
        """Add face embeddings for an employee from uploaded images."""
        try:
            # Handle both UUID and string inputs
            if isinstance(employee_id, str):
                employee_uuid = uuid.UUID(employee_id)
            else:
                employee_uuid = employee_id
            employee = self.db.query(Employee).filter(Employee.id == employee_uuid).first()
            if not employee:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Employee not found"
                )
            
            embeddings = []
            
            for i, image_file in enumerate(image_files):
                # Validate file type
                if not image_file.content_type or not image_file.content_type.startswith('image/'):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"File {image_file.filename} is not a valid image"
                    )
                
                # Save uploaded file
                file_extension = os.path.splitext(image_file.filename)[1]
                filename = f"{str(employee_uuid).replace('-', '')}_{i+1}_{uuid.uuid4().hex}{file_extension}"
                file_path = os.path.join(self.upload_dir, filename)
                
                with open(file_path, "wb") as buffer:
                    content = await image_file.read()
                    buffer.write(content)
                
                # Extract face embedding
                embedding_vector = deepface_service.extract_embedding(file_path)
                
                if embedding_vector is None:
                    # Clean up the saved file
                    os.remove(file_path)
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"No face detected in image {image_file.filename}. Please ensure the image contains a clear face."
                    )
                
                # Create embedding record (convert numpy array to list for JSON serialization)
                db_embedding = EmployeeEmbedding(
                    employee_id=employee_uuid,  # Use the UUID object
                    embedding=json.dumps(embedding_vector.tolist() if hasattr(embedding_vector, 'tolist') else list(embedding_vector)),  # Store as JSON string
                    image_path=file_path
                )
                
                self.db.add(db_embedding)
                embeddings.append(db_embedding)
            
            self.db.commit()
            
            # Refresh all embeddings to get their IDs
            for embedding in embeddings:
                self.db.refresh(embedding)
            
            # Cache the new embeddings in Redis
            if redis_service.is_available():
                cache_start = time.time()
                logger.info(f"🟡 REDIS CACHE - Employee Registration: Starting cache operation for {len(embeddings)} new embeddings")
                
                try:
                    # Prepare embeddings for caching
                    cache_embeddings = []
                    for emb in embeddings:
                        cache_embeddings.append({
                            'id': str(emb.id),
                            'embedding': emb.embedding,
                            'image_path': emb.image_path,
                            'created_at': emb.created_at.isoformat()
                        })
                    
                    # Cache individual employee embeddings
                    cache_success = redis_service.cache_employee_embeddings(str(employee_uuid), cache_embeddings)
                    cache_time = time.time() - cache_start
                    
                    if cache_success:
                        logger.info(f"🟢 REDIS CACHE - Employee Registration: Successfully cached embeddings for employee {employee_uuid} in {cache_time:.3f}s")
                    else:
                        logger.warning(f"🟠 REDIS CACHE - Employee Registration: Failed to cache embeddings for employee {employee_uuid} after {cache_time:.3f}s")
                        
                except Exception as cache_error:
                    cache_time = time.time() - cache_start
                    logger.error(f"🔴 REDIS CACHE - Employee Registration: Cache operation failed for employee {employee_uuid} after {cache_time:.3f}s: {str(cache_error)}")
            else:
                logger.warning(f"🔴 REDIS CACHE - Employee Registration: Redis unavailable, skipping cache for employee {employee_uuid}")
            
            return embeddings
            
        except HTTPException:
            self.db.rollback()
            raise
        except Exception as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error adding embeddings: {str(e)}"
            )
    
    def get_employee(self, employee_id: str) -> Optional[Employee]:
        """Get employee by ID."""
        try:
            employee_uuid = uuid.UUID(employee_id)
            return self.db.query(Employee).filter(Employee.id == employee_uuid).first()
        except ValueError:
            return None
    
    def get_employee_by_email(self, email: str) -> Optional[Employee]:
        """Get employee by email."""
        return self.db.query(Employee).filter(Employee.email == email).first()
    
    def get_all_employees(self, skip: int = 0, limit: int = 100, active_only: bool = True) -> List[Employee]:
        """Get all employees with pagination."""
        query = self.db.query(Employee)
        
        if active_only:
            query = query.filter(Employee.is_active == True)
        
        return query.offset(skip).limit(limit).all()
    
    def update_employee(self, employee_id: str, employee_data: EmployeeUpdate) -> Optional[Employee]:
        """Update employee information."""
        try:
            employee_uuid = uuid.UUID(employee_id)
            employee = self.db.query(Employee).filter(Employee.id == employee_uuid).first()
            
            if not employee:
                return None
            
            # Update fields if provided
            update_data = employee_data.dict(exclude_unset=True)
            
            # Check email uniqueness if email is being updated
            if 'email' in update_data:
                existing_employee = self.db.query(Employee).filter(
                    Employee.email == update_data['email'],
                    Employee.id != employee_uuid
                ).first()
                
                if existing_employee:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Employee with this email already exists"
                    )
            
            for field, value in update_data.items():
                setattr(employee, field, value)
            
            self.db.commit()
            self.db.refresh(employee)
            
            return employee
            
        except IntegrityError:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Employee with this email already exists"
            )
        except Exception as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error updating employee: {str(e)}"
            )
    
    def deactivate_employee(self, employee_id: str) -> bool:
        """Deactivate an employee (soft delete)."""
        try:
            employee_uuid = uuid.UUID(employee_id)
            employee = self.db.query(Employee).filter(Employee.id == employee_uuid).first()
            
            if not employee:
                return False
            
            employee.is_active = False
            self.db.commit()
            
            return True
            
        except Exception as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error deactivating employee: {str(e)}"
            )
    
    def get_employee_embeddings(self, employee_id: str) -> List[EmployeeEmbedding]:
        """Get all embeddings for an employee."""
        try:
            employee_uuid = uuid.UUID(employee_id)
            return self.db.query(EmployeeEmbedding).filter(
                EmployeeEmbedding.employee_id == employee_uuid
            ).all()
        except ValueError:
            return []
    
    def get_all_embeddings(self) -> List[dict]:
        """Get all employee embeddings for face recognition."""
        embeddings = self.db.query(EmployeeEmbedding).join(Employee).filter(
            Employee.is_active == True
        ).all()
        
        return [
            {
                'employee_id': str(embedding.employee_id),
                'embedding': embedding.embedding
            }
            for embedding in embeddings
        ]
    
    async def save_auto_capture_image(self, user_id: str, index: int, image_file: UploadFile) -> dict:
        """Save a single auto-captured image to dataset/{user_id}/ directory."""
        try:
            # Create user-specific dataset directory
            user_dataset_dir = os.path.join(self.dataset_dir, user_id)
            os.makedirs(user_dataset_dir, exist_ok=True)
            
            # Save image with numbered filename
            file_extension = ".jpg"  # Standardize to jpg
            filename = f"{index}.jpg"
            file_path = os.path.join(user_dataset_dir, filename)
            
            # Save the uploaded file
            with open(file_path, "wb") as buffer:
                content = await image_file.read()
                buffer.write(content)
            
            return {
                "file_path": file_path,
                "filename": filename,
                "user_id": user_id,
                "index": index
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error saving auto-capture image: {str(e)}"
            )
    
    async def process_auto_captured_images(self, employee_id: str, user_id: str) -> List[EmployeeEmbedding]:
        """Process all auto-captured images from dataset/{user_id}/ directory."""
        try:
            employee_uuid = uuid.UUID(employee_id)
            user_dataset_dir = os.path.join(self.dataset_dir, user_id)
            
            if not os.path.exists(user_dataset_dir):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"No auto-captured images found for user {user_id}"
                )
            
            # Get all image files in the dataset directory
            image_files = []
            for filename in os.listdir(user_dataset_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(user_dataset_dir, filename)
                    # Extract index from filename (e.g., "0.jpg" -> 0)
                    try:
                        index = int(os.path.splitext(filename)[0])
                        image_files.append((index, file_path))
                    except ValueError:
                        continue
            
            # Sort by index to maintain order
            image_files.sort(key=lambda x: x[0])
            
            if len(image_files) < 3:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Not enough images found. Need at least 3, found {len(image_files)}"
                )
            
            embeddings = []
            processed_count = 0
            
            for index, file_path in image_files:
                try:
                    # Extract face embedding
                    embedding_vector = deepface_service.extract_embedding(file_path)
                    
                    if embedding_vector is None:
                        # Skip images without detectable faces but don't fail completely
                        print(f"Warning: No face detected in {file_path}, skipping...")
                        continue
                    
                    # Move image to permanent storage
                    file_extension = os.path.splitext(file_path)[1]
                    permanent_filename = f"{str(employee_uuid).replace('-', '')}_{index}_{uuid.uuid4().hex}{file_extension}"
                    permanent_path = os.path.join(self.upload_dir, permanent_filename)
                    
                    # Copy file to permanent location
                    import shutil
                    shutil.copy2(file_path, permanent_path)
                    
                    # Create embedding record (convert numpy array to list for JSON serialization)
                    db_embedding = EmployeeEmbedding(
                        employee_id=employee_uuid,
                        embedding=json.dumps(embedding_vector.tolist() if hasattr(embedding_vector, 'tolist') else list(embedding_vector)),
                        image_path=permanent_path
                    )
                    
                    self.db.add(db_embedding)
                    embeddings.append(db_embedding)
                    processed_count += 1
                    
                except Exception as e:
                    print(f"Error processing image {file_path}: {str(e)}")
                    continue
            
            if processed_count < 3:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Not enough valid face images. Processed {processed_count}, need at least 3."
                )
            
            self.db.commit()
            
            # Refresh all embeddings to get their IDs
            for embedding in embeddings:
                self.db.refresh(embedding)
            
            # Clean up temporary dataset directory
            try:
                import shutil
                shutil.rmtree(user_dataset_dir)
            except Exception as e:
                print(f"Warning: Could not clean up temporary directory {user_dataset_dir}: {str(e)}")
            
            return embeddings
            
        except HTTPException:
            self.db.rollback()
            raise
        except Exception as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing auto-captured images: {str(e)}"
            )
    
    # Training Photo Methods
    def start_training_collection(self, employee_id: str) -> dict:
        """Start training photo collection for an employee."""
        try:
            employee_uuid = uuid.UUID(employee_id)
            employee = self.db.query(Employee).filter(Employee.id == employee_uuid).first()
            
            if not employee:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Employee not found"
                )
            
            # Update training status
            employee.training_status = "collecting"
            self.db.commit()
            
            return {
                "employee_id": str(employee.id),
                "name": employee.name,
                "status": "collecting",
                "required_photos": 15,
                "collected_photos": employee.training_photos_count
            }
            
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid employee ID format"
            )
        except Exception as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error starting training collection: {str(e)}"
            )
    
    async def add_training_photo(self, employee_id: str, image_file: UploadFile, 
                                pose_type: str, lighting_condition: str, 
                                expression: str) -> dict:
        """Add a training photo for an employee."""
        try:
            employee_uuid = uuid.UUID(employee_id)
            employee = self.db.query(Employee).filter(Employee.id == employee_uuid).first()
            
            if not employee:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Employee not found"
                )
            
            # Validate file type
            if not image_file.content_type or not image_file.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid image file"
                )
            
            # Save image
            file_extension = os.path.splitext(image_file.filename)[1]
            filename = f"training_{str(employee_uuid).replace('-', '')}_{uuid.uuid4().hex}{file_extension}"
            file_path = os.path.join(self.training_dir, filename)
            
            with open(file_path, "wb") as buffer:
                content = await image_file.read()
                buffer.write(content)
            
            # Calculate quality score using existing face_quality_validator
            quality_result = face_quality_validator.validate_face_quality(file_path)
            quality_score = quality_result.confidence
            
            # Create training photo record
            training_photo = TrainingPhoto(
                employee_id=employee_uuid,
                image_path=file_path,
                pose_type=pose_type,
                quality_score=quality_score,
                lighting_condition=lighting_condition,
                expression=expression,
                is_validated=quality_score > 0.7
            )
            
            self.db.add(training_photo)
            
            # Update employee training stats
            employee.training_photos_count += 1
            employee.last_training_photo = datetime.now()
            
            if employee.training_photos_count >= 15:
                employee.training_status = "completed"
            
            self.db.commit()
            
            return {
                "photo_id": str(training_photo.id),
                "quality_score": quality_score,
                "is_validated": training_photo.is_validated,
                "total_photos": employee.training_photos_count
            }
            
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid employee ID format"
            )
        except Exception as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error adding training photo: {str(e)}"
            )
    
    def get_training_progress(self, employee_id: str) -> dict:
        """Get training photo collection progress."""
        try:
            employee_uuid = uuid.UUID(employee_id)
            employee = self.db.query(Employee).filter(Employee.id == employee_uuid).first()
            
            if not employee:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Employee not found"
                )
            
            photos = self.db.query(TrainingPhoto).filter(
                TrainingPhoto.employee_id == employee_uuid
            ).all()
            
            validated_count = len([p for p in photos if p.is_validated])
            completion_percentage = (len(photos) / 15) * 100 if len(photos) > 0 else 0
            
            return {
                "employee_id": str(employee.id),
                "name": employee.name,
                "status": employee.training_status,
                "total_photos": len(photos),
                "validated_photos": validated_count,
                "required_photos": 15,
                "completion_percentage": completion_percentage,
                "photos": [
                    {
                        "id": str(p.id),
                        "pose_type": p.pose_type,
                        "quality_score": p.quality_score,
                        "is_validated": p.is_validated,
                        "created_at": p.created_at
                    } for p in photos
                ]
            }
            
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid employee ID format"
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error getting training progress: {str(e)}"
            )
