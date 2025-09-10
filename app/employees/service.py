import os
import json
import uuid
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status, UploadFile
from app.employees.models import Employee, EmployeeEmbedding
from app.employees.schemas import EmployeeCreate, EmployeeUpdate
from app.face_recognition.deepface_service import deepface_service


class EmployeeService:
    def __init__(self, db: Session):
        self.db = db
        self.upload_dir = "uploads/employee_images"
        self.dataset_dir = "dataset"
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.dataset_dir, exist_ok=True)
    
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
