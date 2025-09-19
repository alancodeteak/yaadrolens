"""
Example Enhanced Route with Proper Error Handling

This file demonstrates how to implement routes with the new error handling system.
You can use this as a reference when updating existing routes.
"""

from fastapi import APIRouter, Depends, UploadFile, File, Form, Query
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime

from app.core.database import get_db
from app.core.dependencies import get_current_admin_user, get_current_user
from app.auth.models import User
from app.core.exceptions import (
    ResourceNotFoundError,
    ValidationError,
    InvalidFileError,
    FaceQualityError
)
from app.core.validators import (
    validate_uuid,
    validate_email,
    validate_image_file,
    validate_employee_data,
    validate_pagination
)
from app.core.route_decorators import handle_service_errors, log_route_access

# Create router
router = APIRouter(prefix="/example", tags=["example"])


@router.post("/employee")
@handle_service_errors
@log_route_access
async def create_employee_example(
    name: str = Form(...),
    email: str = Form(...),
    department: Optional[str] = Form(None),
    position: Optional[str] = Form(None),
    phone: Optional[str] = Form(None),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Example: Create employee with proper error handling
    
    This demonstrates:
    - Input validation
    - Custom exceptions
    - Service layer error handling
    - Proper logging
    """
    
    # Validate input data
    validate_employee_data(name, email, department, position)
    
    # Create employee using service layer
    # (Service layer handles database errors and business logic)
    from app.employees.service import EmployeeService
    
    employee_service = EmployeeService(db)
    
    # The service will raise appropriate exceptions that are handled by our error handlers
    employee_data = {
        "name": name.strip(),
        "email": validate_email(email),
        "department": department.strip() if department else None,
        "position": position.strip() if position else None,
        "phone": phone.strip() if phone else None
    }
    
    # Convert to Pydantic model for service
    from app.employees.schemas import EmployeeCreate
    employee_create = EmployeeCreate(**employee_data)
    
    employee = employee_service.create_employee(employee_create)
    
    return {
        "success": True,
        "message": "Employee created successfully",
        "employee": {
            "id": str(employee.id),
            "name": employee.name,
            "email": employee.email,
            "department": employee.department,
            "position": employee.position,
            "phone": employee.phone
        }
    }


@router.get("/employee/{employee_id}")
@handle_service_errors
@log_route_access
async def get_employee_example(
    employee_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Example: Get employee with proper error handling
    
    This demonstrates:
    - UUID validation
    - Resource not found handling
    - Proper response structure
    """
    
    # Validate UUID format
    validate_uuid(employee_id, "employee_id")
    
    # Get employee using service layer
    from app.employees.service import EmployeeService
    
    employee_service = EmployeeService(db)
    
    # This will raise ResourceNotFoundError if employee doesn't exist
    employee = employee_service.get_employee(employee_id)
    
    if not employee:
        raise ResourceNotFoundError("Employee", employee_id)
    
    return {
        "success": True,
        "employee": {
            "id": str(employee.id),
            "name": employee.name,
            "email": employee.email,
            "department": employee.department,
            "position": employee.position,
            "phone": employee.phone,
            "is_active": employee.is_active,
            "created_at": employee.created_at.isoformat(),
            "updated_at": employee.updated_at.isoformat() if employee.updated_at else None
        }
    }


@router.get("/employees")
@handle_service_errors
@log_route_access
async def list_employees_example(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    active_only: bool = Query(True),
    department: Optional[str] = Query(None),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Example: List employees with pagination and filtering
    
    This demonstrates:
    - Pagination validation
    - Query parameter validation
    - Filtering
    """
    
    # Validate pagination parameters
    validate_pagination(skip, limit)
    
    # Get employees using service layer
    from app.employees.service import EmployeeService
    
    employee_service = EmployeeService(db)
    employees = employee_service.get_all_employees(
        skip=skip,
        limit=limit,
        active_only=active_only
    )
    
    # Filter by department if provided
    if department:
        employees = [emp for emp in employees if emp.department == department]
    
    return {
        "success": True,
        "count": len(employees),
        "pagination": {
            "skip": skip,
            "limit": limit,
            "total_returned": len(employees)
        },
        "filters": {
            "active_only": active_only,
            "department": department
        },
        "employees": [
            {
                "id": str(emp.id),
                "name": emp.name,
                "email": emp.email,
                "department": emp.department,
                "position": emp.position,
                "is_active": emp.is_active
            }
            for emp in employees
        ]
    }


@router.post("/employee/{employee_id}/photo")
@handle_service_errors
@log_route_access
async def upload_employee_photo_example(
    employee_id: str,
    photo: UploadFile = File(...),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Example: Upload employee photo with file validation
    
    This demonstrates:
    - File validation
    - Image processing error handling
    - Custom business logic errors
    """
    
    # Validate UUID format
    validate_uuid(employee_id, "employee_id")
    
    # Validate image file
    validate_image_file(photo, max_size=5 * 1024 * 1024)  # 5MB max
    
    # Check if employee exists
    from app.employees.service import EmployeeService
    
    employee_service = EmployeeService(db)
    employee = employee_service.get_employee(employee_id)
    
    if not employee:
        raise ResourceNotFoundError("Employee", employee_id)
    
    if not employee.is_active:
        from app.core.exceptions import ResourceInactiveError
        raise ResourceInactiveError("Employee", employee_id)
    
    # Process the image (this is where face quality checks would happen)
    try:
        # Simulate face quality validation
        # In real implementation, this would use face recognition service
        
        # Read file content for processing
        content = await photo.read()
        
        # Reset file pointer for potential re-use
        await photo.seek(0)
        
        # Simulate face quality check
        if len(content) < 1000:  # Simulate poor quality image
            raise FaceQualityError(
                detail="Image quality is too low for face recognition",
                quality_issues=["low_resolution", "insufficient_data"],
                error_data={"file_size": len(content)}
            )
        
        # Save the photo (in real implementation)
        # photo_path = await employee_service.save_employee_photo(employee_id, photo)
        
        return {
            "success": True,
            "message": "Employee photo uploaded successfully",
            "employee_id": employee_id,
            "file_info": {
                "filename": photo.filename,
                "content_type": photo.content_type,
                "size": len(content)
            }
        }
        
    except Exception as e:
        # Log the error
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error processing employee photo: {str(e)}")
        
        # Re-raise as appropriate exception
        if "face" in str(e).lower():
            raise FaceQualityError(f"Face processing error: {str(e)}")
        else:
            raise ValidationError(f"Image processing error: {str(e)}")


@router.delete("/employee/{employee_id}")
@handle_service_errors
@log_route_access
async def delete_employee_example(
    employee_id: str,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Example: Delete (deactivate) employee
    
    This demonstrates:
    - Soft delete operations
    - Audit logging
    - Success responses
    """
    
    # Validate UUID format
    validate_uuid(employee_id, "employee_id")
    
    # Deactivate employee using service layer
    from app.employees.service import EmployeeService
    
    employee_service = EmployeeService(db)
    success = employee_service.deactivate_employee(employee_id)
    
    if not success:
        raise ResourceNotFoundError("Employee", employee_id)
    
    return {
        "success": True,
        "message": "Employee deactivated successfully",
        "employee_id": employee_id,
        "deactivated_by": str(current_user.id),
        "deactivated_at": datetime.utcnow().isoformat()
    }


# Error handling examples for different scenarios

@router.get("/error-examples/validation")
async def validation_error_example():
    """Example endpoint that demonstrates validation errors."""
    # This will trigger a validation error
    validate_email("invalid-email")


@router.get("/error-examples/not-found")
async def not_found_error_example():
    """Example endpoint that demonstrates not found errors."""
    # This will trigger a not found error
    raise ResourceNotFoundError("TestResource", "non-existent-id")


@router.get("/error-examples/database")
async def database_error_example():
    """Example endpoint that demonstrates database errors."""
    # This will trigger a database error
    from app.core.exceptions import DatabaseError
    raise DatabaseError("Simulated database connection error")


@router.post("/error-examples/file-upload")
async def file_upload_error_example(file: UploadFile = File(...)):
    """Example endpoint that demonstrates file upload errors."""
    # This will trigger file validation errors
    validate_image_file(file, max_size=1024)  # Very small limit to trigger error

