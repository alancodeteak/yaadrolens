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
