"""
Custom Exception Classes for the Face Recognition Attendance System
"""

from typing import Any, Dict, Optional
from fastapi import HTTPException, status


class BaseAPIException(HTTPException):
    """Base exception class for API errors with enhanced error details."""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: Optional[str] = None,
        error_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code
        self.error_data = error_data or {}


# Authentication & Authorization Exceptions
class AuthenticationError(BaseAPIException):
    """Authentication failed."""
    
    def __init__(self, detail: str = "Authentication failed", error_data: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            error_code="AUTH_FAILED",
            error_data=error_data,
            headers={"WWW-Authenticate": "Bearer"}
        )


class InvalidTokenError(BaseAPIException):
    """Invalid or expired token."""
    
    def __init__(self, detail: str = "Invalid or expired token", error_data: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            error_code="INVALID_TOKEN",
            error_data=error_data,
            headers={"WWW-Authenticate": "Bearer"}
        )


class InsufficientPermissionsError(BaseAPIException):
    """User doesn't have required permissions."""
    
    def __init__(self, detail: str = "Insufficient permissions", error_data: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            error_code="INSUFFICIENT_PERMISSIONS",
            error_data=error_data
        )


# Resource Exceptions
class ResourceNotFoundError(BaseAPIException):
    """Requested resource not found."""
    
    def __init__(self, resource_type: str, resource_id: str = None, error_data: Optional[Dict[str, Any]] = None):
        detail = f"{resource_type} not found"
        if resource_id:
            detail += f" (ID: {resource_id})"
        
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
            error_code="RESOURCE_NOT_FOUND",
            error_data={"resource_type": resource_type, "resource_id": resource_id, **(error_data or {})}
        )


class ResourceAlreadyExistsError(BaseAPIException):
    """Resource already exists."""
    
    def __init__(self, resource_type: str, field: str = None, value: str = None, error_data: Optional[Dict[str, Any]] = None):
        detail = f"{resource_type} already exists"
        if field and value:
            detail += f" with {field}: {value}"
        
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail,
            error_code="RESOURCE_ALREADY_EXISTS",
            error_data={"resource_type": resource_type, "field": field, "value": value, **(error_data or {})}
        )


class ResourceInactiveError(BaseAPIException):
    """Resource is inactive or disabled."""
    
    def __init__(self, resource_type: str, resource_id: str = None, error_data: Optional[Dict[str, Any]] = None):
        detail = f"{resource_type} is inactive"
        if resource_id:
            detail += f" (ID: {resource_id})"
        
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code="RESOURCE_INACTIVE",
            error_data={"resource_type": resource_type, "resource_id": resource_id, **(error_data or {})}
        )


# Validation Exceptions
class ValidationError(BaseAPIException):
    """Data validation failed."""
    
    def __init__(self, detail: str, field: str = None, value: Any = None, error_data: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            error_code="VALIDATION_ERROR",
            error_data={"field": field, "value": value, **(error_data or {})}
        )


class InvalidFileError(BaseAPIException):
    """Invalid file uploaded."""
    
    def __init__(self, detail: str = "Invalid file", file_type: str = None, error_data: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code="INVALID_FILE",
            error_data={"file_type": file_type, **(error_data or {})}
        )


class FileTooLargeError(BaseAPIException):
    """File size exceeds limit."""
    
    def __init__(self, max_size: int, actual_size: int = None, error_data: Optional[Dict[str, Any]] = None):
        detail = f"File size exceeds maximum limit of {max_size} bytes"
        if actual_size:
            detail += f" (uploaded: {actual_size} bytes)"
        
        super().__init__(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=detail,
            error_code="FILE_TOO_LARGE",
            error_data={"max_size": max_size, "actual_size": actual_size, **(error_data or {})}
        )


# Business Logic Exceptions
class FaceRecognitionError(BaseAPIException):
    """Face recognition processing failed."""
    
    def __init__(self, detail: str = "Face recognition failed", error_data: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            error_code="FACE_RECOGNITION_ERROR",
            error_data=error_data
        )


class FaceQualityError(BaseAPIException):
    """Face image quality is insufficient."""
    
    def __init__(self, detail: str = "Face image quality is insufficient", quality_issues: list = None, error_data: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code="FACE_QUALITY_ERROR",
            error_data={"quality_issues": quality_issues or [], **(error_data or {})}
        )


class NoFaceDetectedError(BaseAPIException):
    """No face detected in the image."""
    
    def __init__(self, detail: str = "No face detected in the image", error_data: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code="NO_FACE_DETECTED",
            error_data=error_data
        )


class MultipleFacesDetectedError(BaseAPIException):
    """Multiple faces detected in the image."""
    
    def __init__(self, face_count: int = None, error_data: Optional[Dict[str, Any]] = None):
        detail = "Multiple faces detected in the image"
        if face_count:
            detail += f" ({face_count} faces found)"
        
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code="MULTIPLE_FACES_DETECTED",
            error_data={"face_count": face_count, **(error_data or {})}
        )


class EmployeeNotEnrolledError(BaseAPIException):
    """Employee not enrolled for face recognition."""
    
    def __init__(self, employee_id: str = None, error_data: Optional[Dict[str, Any]] = None):
        detail = "Employee not enrolled for face recognition"
        if employee_id:
            detail += f" (Employee ID: {employee_id})"
        
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code="EMPLOYEE_NOT_ENROLLED",
            error_data={"employee_id": employee_id, **(error_data or {})}
        )


# Database Exceptions
class DatabaseError(BaseAPIException):
    """Database operation failed."""
    
    def __init__(self, detail: str = "Database operation failed", operation: str = None, error_data: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code="DATABASE_ERROR",
            error_data={"operation": operation, **(error_data or {})}
        )


class DatabaseConnectionError(BaseAPIException):
    """Database connection failed."""
    
    def __init__(self, detail: str = "Database connection failed", error_data: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            error_code="DATABASE_CONNECTION_ERROR",
            error_data=error_data
        )


# External Service Exceptions
class ExternalServiceError(BaseAPIException):
    """External service error."""
    
    def __init__(self, service_name: str, detail: str = None, error_data: Optional[Dict[str, Any]] = None):
        detail = detail or f"{service_name} service error"
        
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            error_code="EXTERNAL_SERVICE_ERROR",
            error_data={"service_name": service_name, **(error_data or {})}
        )


class RedisConnectionError(BaseAPIException):
    """Redis connection failed."""
    
    def __init__(self, detail: str = "Redis service unavailable", error_data: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            error_code="REDIS_CONNECTION_ERROR",
            error_data=error_data
        )


# Rate Limiting Exceptions
class RateLimitExceededError(BaseAPIException):
    """Rate limit exceeded."""
    
    def __init__(self, detail: str = "Rate limit exceeded", retry_after: int = None, error_data: Optional[Dict[str, Any]] = None):
        headers = {}
        if retry_after:
            headers["Retry-After"] = str(retry_after)
        
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            error_code="RATE_LIMIT_EXCEEDED",
            error_data={"retry_after": retry_after, **(error_data or {})},
            headers=headers
        )


# Configuration Exceptions
class ConfigurationError(BaseAPIException):
    """Configuration error."""
    
    def __init__(self, detail: str = "Configuration error", config_key: str = None, error_data: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code="CONFIGURATION_ERROR",
            error_data={"config_key": config_key, **(error_data or {})}
        )


# Payroll Exceptions
class PayrollCalculationError(BaseAPIException):
    """Payroll calculation failed."""
    
    def __init__(self, detail: str = "Payroll calculation failed", employee_id: str = None, error_data: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            error_code="PAYROLL_CALCULATION_ERROR",
            error_data={"employee_id": employee_id, **(error_data or {})}
        )


class InsufficientAttendanceDataError(BaseAPIException):
    """Insufficient attendance data for payroll calculation."""
    
    def __init__(self, employee_id: str, period: str = None, error_data: Optional[Dict[str, Any]] = None):
        detail = f"Insufficient attendance data for employee {employee_id}"
        if period:
            detail += f" for period {period}"
        
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code="INSUFFICIENT_ATTENDANCE_DATA",
            error_data={"employee_id": employee_id, "period": period, **(error_data or {})}
        )


# ML Monitoring Exceptions
class MLModelError(BaseAPIException):
    """ML model operation failed."""
    
    def __init__(self, detail: str = "ML model error", model_name: str = None, error_data: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code="ML_MODEL_ERROR",
            error_data={"model_name": model_name, **(error_data or {})}
        )


class ModelNotLoadedError(BaseAPIException):
    """ML model not loaded."""
    
    def __init__(self, model_name: str = None, error_data: Optional[Dict[str, Any]] = None):
        detail = "ML model not loaded"
        if model_name:
            detail += f": {model_name}"
        
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            error_code="MODEL_NOT_LOADED",
            error_data={"model_name": model_name, **(error_data or {})}
        )


class ThresholdOptimizationError(BaseAPIException):
    """Threshold optimization failed."""
    
    def __init__(self, detail: str = "Threshold optimization failed", error_data: Optional[Dict[str, Any]] = None):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            error_code="THRESHOLD_OPTIMIZATION_ERROR",
            error_data=error_data
        )

