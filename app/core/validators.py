"""
Validation Utilities for the Face Recognition Attendance System
"""

import re
import uuid
from typing import Any, List, Optional
from datetime import datetime, date
from fastapi import UploadFile

from app.core.exceptions import (
    ValidationError,
    InvalidFileError,
    FileTooLargeError
)


def validate_uuid(value: str, field_name: str = "id") -> str:
    """Validate UUID format."""
    try:
        uuid.UUID(value)
        return value
    except (ValueError, TypeError):
        raise ValidationError(
            detail=f"Invalid UUID format for {field_name}",
            field=field_name,
            value=value
        )


def validate_email(email: str) -> str:
    """Validate email format."""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not email or not isinstance(email, str):
        raise ValidationError(
            detail="Email is required",
            field="email",
            value=email
        )
    
    if not re.match(email_pattern, email.strip()):
        raise ValidationError(
            detail="Invalid email format",
            field="email",
            value=email
        )
    
    return email.strip().lower()


def validate_password(password: str, min_length: int = 8) -> str:
    """Validate password strength."""
    if not password or not isinstance(password, str):
        raise ValidationError(
            detail="Password is required",
            field="password"
        )
    
    if len(password) < min_length:
        raise ValidationError(
            detail=f"Password must be at least {min_length} characters long",
            field="password",
            error_data={"min_length": min_length, "actual_length": len(password)}
        )
    
    # Check for at least one uppercase, lowercase, digit, and special character
    if not re.search(r'[A-Z]', password):
        raise ValidationError(
            detail="Password must contain at least one uppercase letter",
            field="password"
        )
    
    if not re.search(r'[a-z]', password):
        raise ValidationError(
            detail="Password must contain at least one lowercase letter",
            field="password"
        )
    
    if not re.search(r'\d', password):
        raise ValidationError(
            detail="Password must contain at least one digit",
            field="password"
        )
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        raise ValidationError(
            detail="Password must contain at least one special character",
            field="password"
        )
    
    return password


def validate_phone(phone: str) -> Optional[str]:
    """Validate phone number format."""
    if not phone:
        return None
    
    # Remove all non-digit characters
    digits_only = re.sub(r'\D', '', phone)
    
    # Check length (assuming international format)
    if len(digits_only) < 10 or len(digits_only) > 15:
        raise ValidationError(
            detail="Phone number must be between 10 and 15 digits",
            field="phone",
            value=phone
        )
    
    return phone.strip()


def validate_date_range(start_date: datetime, end_date: datetime):
    """Validate date range."""
    if start_date >= end_date:
        raise ValidationError(
            detail="Start date must be before end date",
            error_data={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            }
        )
    
    # Check if date range is not too large (e.g., max 1 year)
    max_days = 365
    if (end_date - start_date).days > max_days:
        raise ValidationError(
            detail=f"Date range cannot exceed {max_days} days",
            error_data={
                "max_days": max_days,
                "actual_days": (end_date - start_date).days
            }
        )


def validate_pagination(skip: int, limit: int, max_limit: int = 1000):
    """Validate pagination parameters."""
    if skip < 0:
        raise ValidationError(
            detail="Skip parameter cannot be negative",
            field="skip",
            value=skip
        )
    
    if limit <= 0:
        raise ValidationError(
            detail="Limit parameter must be positive",
            field="limit",
            value=limit
        )
    
    if limit > max_limit:
        raise ValidationError(
            detail=f"Limit parameter cannot exceed {max_limit}",
            field="limit",
            value=limit,
            error_data={"max_limit": max_limit}
        )


def validate_image_file(
    file: UploadFile,
    max_size: int = 10 * 1024 * 1024,  # 10MB
    allowed_formats: List[str] = None
) -> UploadFile:
    """Validate uploaded image file."""
    if not file:
        raise InvalidFileError("No file provided")
    
    if not file.filename:
        raise InvalidFileError("File must have a name")
    
    # Set default allowed formats
    if allowed_formats is None:
        allowed_formats = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp']
    
    # Check file type
    if file.content_type not in allowed_formats:
        raise InvalidFileError(
            detail=f"Invalid file format. Allowed formats: {', '.join(allowed_formats)}",
            file_type=file.content_type,
            error_data={"allowed_formats": allowed_formats}
        )
    
    # Check file size if available
    if hasattr(file, 'size') and file.size:
        if file.size > max_size:
            raise FileTooLargeError(
                max_size=max_size,
                actual_size=file.size
            )
    
    # Check file extension
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    file_extension = file.filename.lower().split('.')[-1]
    if f'.{file_extension}' not in allowed_extensions:
        raise InvalidFileError(
            detail=f"Invalid file extension. Allowed extensions: {', '.join(allowed_extensions)}",
            error_data={"file_extension": file_extension, "allowed_extensions": allowed_extensions}
        )
    
    return file


def validate_employee_data(name: str, email: str, department: str = None, position: str = None):
    """Validate employee data."""
    errors = {}
    
    # Validate name
    if not name or not name.strip():
        errors["name"] = "Name is required"
    elif len(name.strip()) < 2:
        errors["name"] = "Name must be at least 2 characters long"
    elif len(name.strip()) > 100:
        errors["name"] = "Name cannot exceed 100 characters"
    
    # Validate email
    try:
        validate_email(email)
    except ValidationError as e:
        errors["email"] = e.detail
    
    # Validate optional fields
    if department and len(department.strip()) > 50:
        errors["department"] = "Department cannot exceed 50 characters"
    
    if position and len(position.strip()) > 50:
        errors["position"] = "Position cannot exceed 50 characters"
    
    if errors:
        raise ValidationError(
            detail="Employee data validation failed",
            error_data={"validation_errors": errors}
        )


def validate_payroll_data(
    employee_id: str,
    month: int,
    year: int,
    hourly_rate: float = None,
    overtime_rate: float = None,
    deductions: float = None
):
    """Validate payroll data."""
    errors = {}
    
    # Validate employee ID
    try:
        validate_uuid(employee_id, "employee_id")
    except ValidationError as e:
        errors["employee_id"] = e.detail
    
    # Validate month
    if not isinstance(month, int) or month < 1 or month > 12:
        errors["month"] = "Month must be between 1 and 12"
    
    # Validate year
    current_year = datetime.now().year
    if not isinstance(year, int) or year < 2020 or year > current_year + 1:
        errors["year"] = f"Year must be between 2020 and {current_year + 1}"
    
    # Validate rates and deductions
    if hourly_rate is not None:
        if not isinstance(hourly_rate, (int, float)) or hourly_rate < 0:
            errors["hourly_rate"] = "Hourly rate must be a positive number"
        elif hourly_rate > 1000:
            errors["hourly_rate"] = "Hourly rate cannot exceed 1000"
    
    if overtime_rate is not None:
        if not isinstance(overtime_rate, (int, float)) or overtime_rate < 0:
            errors["overtime_rate"] = "Overtime rate must be a positive number"
        elif overtime_rate > 1500:
            errors["overtime_rate"] = "Overtime rate cannot exceed 1500"
    
    if deductions is not None:
        if not isinstance(deductions, (int, float)) or deductions < 0:
            errors["deductions"] = "Deductions must be a positive number"
        elif deductions > 10000:
            errors["deductions"] = "Deductions cannot exceed 10000"
    
    if errors:
        raise ValidationError(
            detail="Payroll data validation failed",
            error_data={"validation_errors": errors}
        )


def validate_attendance_date_range(start_date: datetime, end_date: datetime):
    """Validate attendance date range with specific business rules."""
    validate_date_range(start_date, end_date)
    
    # Additional business rules for attendance
    current_date = datetime.now()
    
    # Don't allow future dates beyond today
    if start_date.date() > current_date.date():
        raise ValidationError(
            detail="Start date cannot be in the future",
            field="start_date",
            value=start_date.isoformat()
        )
    
    if end_date.date() > current_date.date():
        raise ValidationError(
            detail="End date cannot be in the future",
            field="end_date",
            value=end_date.isoformat()
        )


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    if not filename:
        return "unnamed_file"
    
    # Remove path components
    filename = filename.split('/')[-1].split('\\')[-1]
    
    # Remove or replace dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Ensure filename is not empty
    if not filename:
        return "unnamed_file"
    
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        max_name_length = 250 - len(ext)
        filename = name[:max_name_length] + ('.' + ext if ext else '')
    
    return filename

