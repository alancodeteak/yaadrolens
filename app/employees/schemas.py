from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime
from uuid import UUID


class EmployeeCreate(BaseModel):
    name: str
    email: EmailStr
    department: Optional[str] = None
    position: Optional[str] = None
    phone: Optional[str] = None


class EmployeeUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    department: Optional[str] = None
    position: Optional[str] = None
    phone: Optional[str] = None
    is_active: Optional[bool] = None


class EmployeeResponse(BaseModel):
    id: UUID
    name: str
    email: str
    department: Optional[str]
    position: Optional[str]
    phone: Optional[str]
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime]
    # Training fields
    training_photos_count: Optional[int] = 0
    training_quality_score: Optional[float] = 0.0
    training_status: Optional[str] = "pending"
    last_training_photo: Optional[datetime] = None

    class Config:
        from_attributes = True


class EmployeeEmbeddingResponse(BaseModel):
    id: UUID
    employee_id: UUID
    image_path: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class EmployeeWithEmbeddings(EmployeeResponse):
    embeddings: List[EmployeeEmbeddingResponse] = []


class EmployeeEnrollment(BaseModel):
    employee_data: EmployeeCreate
    # Note: Images will be handled as multipart form data in the endpoint


# Training Photo Schemas
class TrainingPhotoResponse(BaseModel):
    id: UUID
    employee_id: UUID
    image_path: str
    pose_type: Optional[str]
    quality_score: Optional[float]
    lighting_condition: Optional[str]
    expression: Optional[str]
    is_validated: bool
    created_at: datetime

    class Config:
        from_attributes = True


class TrainingPhotoCreate(BaseModel):
    pose_type: str
    lighting_condition: str
    expression: str


class TrainingProgressResponse(BaseModel):
    employee_id: str
    name: str
    status: str
    total_photos: int
    validated_photos: int
    required_photos: int
    completion_percentage: float
    photos: List[TrainingPhotoResponse]


class TrainingCollectionStartResponse(BaseModel):
    employee_id: str
    name: str
    status: str
    required_photos: int
    collected_photos: int


class TrainingPhotoAddResponse(BaseModel):
    photo_id: str
    quality_score: float
    is_validated: bool
    total_photos: int
