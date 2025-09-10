from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
from app.core.database import get_db
from app.core.dependencies import get_current_admin_user
from app.auth.models import User
from app.employees.schemas import (
    EmployeeCreate, 
    EmployeeUpdate, 
    EmployeeResponse, 
    EmployeeWithEmbeddings,
    EmployeeEnrollment
)
from app.employees.service import EmployeeService
import json

router = APIRouter(prefix="/employees", tags=["employees"])


@router.post("/", response_model=EmployeeResponse, status_code=status.HTTP_201_CREATED)
async def create_employee(
    employee_data: EmployeeCreate,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Create a new employee."""
    employee_service = EmployeeService(db)
    employee = employee_service.create_employee(employee_data)
    return employee


@router.post("/enroll", response_model=EmployeeWithEmbeddings, status_code=status.HTTP_201_CREATED)
async def enroll_employee(
    name: str = Form(...),
    email: str = Form(...),
    department: Optional[str] = Form(None),
    position: Optional[str] = Form(None),
    phone: Optional[str] = Form(None),
    images: List[UploadFile] = File(..., min_items=3, max_items=30),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Enroll a new employee with face images for recognition."""
    # Validate number of images (updated for auto-capture)
    if len(images) < 3 or len(images) > 30:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please provide between 3 and 30 images"
        )
    
    # Create employee data
    employee_data = EmployeeCreate(
        name=name,
        email=email,
        department=department,
        position=position,
        phone=phone
    )
    
    employee_service = EmployeeService(db)
    
    # Create employee
    employee = employee_service.create_employee(employee_data)
    
    # Add embeddings
    try:
        embeddings = await employee_service.add_employee_embeddings(str(employee.id), images)
        
        # Return employee with embeddings
        return EmployeeWithEmbeddings(
            id=str(employee.id),
            name=employee.name,
            email=employee.email,
            department=employee.department,
            position=employee.position,
            phone=employee.phone,
            is_active=employee.is_active,
            created_at=employee.created_at,
            updated_at=employee.updated_at,
            embeddings=[
                {
                    "id": str(emb.id),
                    "employee_id": str(emb.employee_id),
                    "image_path": emb.image_path,
                    "created_at": emb.created_at
                }
                for emb in embeddings
            ]
        )
    except Exception as e:
        # If embedding creation fails, clean up the employee
        employee_service.deactivate_employee(str(employee.id))
        raise e


@router.post("/register_face", status_code=status.HTTP_201_CREATED)
async def register_face_auto(
    user_id: str = Form(...),
    index: int = Form(...),
    image: UploadFile = File(...),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Register a single face image for auto-capture enrollment."""
    try:
        # Validate file type
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image file"
            )
        
        employee_service = EmployeeService(db)
        result = await employee_service.save_auto_capture_image(user_id, index, image)
        
        return {
            "success": True,
            "message": f"Image {index} saved successfully",
            "file_path": result["file_path"]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving image: {str(e)}"
        )


@router.post("/finalize_registration", response_model=EmployeeWithEmbeddings, status_code=status.HTTP_201_CREATED)
async def finalize_auto_registration(
    user_id: str = Form(...),
    name: str = Form(...),
    email: str = Form(...),
    department: Optional[str] = Form(None),
    position: Optional[str] = Form(None),
    phone: Optional[str] = Form(None),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Finalize auto-capture registration by processing all saved images."""
    try:
        # Create employee data
        employee_data = EmployeeCreate(
            name=name,
            email=email,
            department=department,
            position=position,
            phone=phone
        )
        
        employee_service = EmployeeService(db)
        
        # Create employee
        employee = employee_service.create_employee(employee_data)
        
        # Process auto-captured images
        embeddings = await employee_service.process_auto_captured_images(str(employee.id), user_id)
        
        # Return employee with embeddings
        return EmployeeWithEmbeddings(
            id=str(employee.id),
            name=employee.name,
            email=employee.email,
            department=employee.department,
            position=employee.position,
            phone=employee.phone,
            is_active=employee.is_active,
            created_at=employee.created_at,
            updated_at=employee.updated_at,
            embeddings=[
                {
                    "id": str(emb.id),
                    "employee_id": str(emb.employee_id),
                    "image_path": emb.image_path,
                    "created_at": emb.created_at
                }
                for emb in embeddings
            ]
        )
        
    except Exception as e:
        # If processing fails, clean up the employee
        if 'employee' in locals():
            employee_service.deactivate_employee(str(employee.id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error finalizing registration: {str(e)}"
        )


@router.get("/", response_model=List[EmployeeResponse])
async def get_employees(
    skip: int = 0,
    limit: int = 100,
    active_only: bool = True,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get all employees with pagination."""
    employee_service = EmployeeService(db)
    employees = employee_service.get_all_employees(skip=skip, limit=limit, active_only=active_only)
    return employees


@router.get("/{employee_id}", response_model=EmployeeWithEmbeddings)
async def get_employee(
    employee_id: str,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get employee by ID with embeddings."""
    employee_service = EmployeeService(db)
    employee = employee_service.get_employee(employee_id)
    
    if not employee:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Employee not found"
        )
    
    embeddings = employee_service.get_employee_embeddings(employee_id)
    
    return EmployeeWithEmbeddings(
        id=str(employee.id),
        name=employee.name,
        email=employee.email,
        department=employee.department,
        position=employee.position,
        phone=employee.phone,
        is_active=employee.is_active,
        created_at=employee.created_at,
        updated_at=employee.updated_at,
        embeddings=[
            {
                "id": str(emb.id),
                "employee_id": str(emb.employee_id),
                "image_path": emb.image_path,
                "created_at": emb.created_at
            }
            for emb in embeddings
        ]
    )


@router.put("/{employee_id}", response_model=EmployeeResponse)
async def update_employee(
    employee_id: str,
    employee_data: EmployeeUpdate,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Update employee information."""
    employee_service = EmployeeService(db)
    employee = employee_service.update_employee(employee_id, employee_data)
    
    if not employee:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Employee not found"
        )
    
    return employee


@router.delete("/{employee_id}", status_code=status.HTTP_204_NO_CONTENT)
async def deactivate_employee(
    employee_id: str,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Deactivate an employee (soft delete)."""
    employee_service = EmployeeService(db)
    success = employee_service.deactivate_employee(employee_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Employee not found"
        )


@router.post("/{employee_id}/embeddings", status_code=status.HTTP_201_CREATED)
async def add_employee_embeddings(
    employee_id: str,
    images: List[UploadFile] = File(..., min_items=1, max_items=5),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Add additional face embeddings for an existing employee."""
    employee_service = EmployeeService(db)
    
    # Check if employee exists
    employee = employee_service.get_employee(employee_id)
    if not employee:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Employee not found"
        )
    
    embeddings = await employee_service.add_employee_embeddings(employee_id, images)
    
    return {
        "message": f"Successfully added {len(embeddings)} embeddings",
        "embeddings": [
            {
                "id": str(emb.id),
                "employee_id": str(emb.employee_id),
                "image_path": emb.image_path,
                "created_at": emb.created_at
            }
            for emb in embeddings
        ]
    }
