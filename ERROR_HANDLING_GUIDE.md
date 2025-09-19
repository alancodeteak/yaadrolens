# Error Handling Guide

This document explains the comprehensive error handling system implemented in the Face Recognition Attendance System API.

## Overview

The error handling system provides:

- **Standardized error responses** across all endpoints
- **Custom exception classes** for different error types
- **Automatic error logging** and monitoring
- **Request tracking** with unique IDs
- **Security headers** and middleware protection
- **Rate limiting** and request size validation

## Error Response Format

All API errors now return a standardized JSON format:

```json
{
  "error": true,
  "status_code": 400,
  "detail": "Human-readable error message",
  "error_code": "VALIDATION_ERROR",
  "error_data": {
    "field": "email",
    "value": "invalid-email"
  },
  "timestamp": "2023-10-01T12:00:00.000Z",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Response Fields

- **error**: Always `true` for error responses
- **status_code**: HTTP status code
- **detail**: Human-readable error description
- **error_code**: Machine-readable error identifier
- **error_data**: Additional context about the error
- **timestamp**: When the error occurred
- **request_id**: Unique identifier for request tracking

## Custom Exception Classes

### Base Exception

All custom exceptions inherit from `BaseAPIException`:

```python
from app.core.exceptions import BaseAPIException

raise BaseAPIException(
    status_code=400,
    detail="Custom error message",
    error_code="CUSTOM_ERROR",
    error_data={"key": "value"}
)
```

### Authentication & Authorization

```python
from app.core.exceptions import (
    AuthenticationError,
    InvalidTokenError,
    InsufficientPermissionsError
)

# Invalid credentials
raise AuthenticationError("Invalid email or password")

# Expired/invalid token
raise InvalidTokenError("Token has expired")

# Insufficient permissions
raise InsufficientPermissionsError("Admin access required")
```

### Resource Management

```python
from app.core.exceptions import (
    ResourceNotFoundError,
    ResourceAlreadyExistsError,
    ResourceInactiveError
)

# Resource not found
raise ResourceNotFoundError("Employee", "employee-id-123")

# Duplicate resource
raise ResourceAlreadyExistsError("User", "email", "user@example.com")

# Inactive resource
raise ResourceInactiveError("Employee", "employee-id-123")
```

### Validation Errors

```python
from app.core.exceptions import (
    ValidationError,
    InvalidFileError,
    FileTooLargeError
)

# General validation error
raise ValidationError("Invalid input data", field="name", value="")

# File validation
raise InvalidFileError("Invalid file type", file_type="text/plain")

# File size error
raise FileTooLargeError(max_size=5*1024*1024, actual_size=10*1024*1024)
```

### Business Logic Errors

```python
from app.core.exceptions import (
    FaceRecognitionError,
    FaceQualityError,
    NoFaceDetectedError,
    MultipleFacesDetectedError
)

# Face recognition failed
raise FaceRecognitionError("Unable to process face image")

# Poor image quality
raise FaceQualityError(
    "Image quality insufficient",
    quality_issues=["low_resolution", "poor_lighting"]
)

# No face in image
raise NoFaceDetectedError()

# Multiple faces detected
raise MultipleFacesDetectedError(face_count=3)
```

### Database Errors

```python
from app.core.exceptions import (
    DatabaseError,
    DatabaseConnectionError
)

# General database error
raise DatabaseError("Database operation failed", operation="insert")

# Connection error
raise DatabaseConnectionError("Unable to connect to database")
```

## Using the Service Base Class

Extend `BaseService` for consistent error handling:

```python
from app.core.service_base import BaseService
from app.core.exceptions import ResourceNotFoundError

class MyService(BaseService):
    def __init__(self, db: Session):
        super().__init__(db)
    
    def get_resource(self, resource_id: str):
        # Use built-in error handling methods
        return self.get_or_404(MyModel, resource_id, "MyResource")
    
    def create_resource(self, data: dict):
        # Validate required fields
        self.validate_required_fields(data, ["name", "email"])
        
        # Check uniqueness
        self.check_unique_constraint(MyModel, "email", data["email"])
        
        # Create resource
        resource = MyModel(**data)
        self.db.add(resource)
        self.safe_commit("Error creating resource")
        
        # Log action
        self.log_service_action("create_resource", "MyResource", str(resource.id))
        
        return resource
```

## Validation Utilities

Use the validation utilities for consistent input validation:

```python
from app.core.validators import (
    validate_uuid,
    validate_email,
    validate_image_file,
    validate_employee_data,
    validate_pagination
)

# UUID validation
employee_id = validate_uuid(employee_id, "employee_id")

# Email validation
email = validate_email("user@example.com")

# File validation
validate_image_file(uploaded_file, max_size=5*1024*1024)

# Complex validation
validate_employee_data(name, email, department, position)

# Pagination validation
validate_pagination(skip=0, limit=100)
```

## Route Decorators

Use decorators to enhance route error handling:

```python
from app.core.route_decorators import (
    handle_service_errors,
    log_route_access,
    require_admin
)

@router.post("/employees")
@handle_service_errors  # Handles common service errors
@log_route_access      # Logs route access for auditing
@require_admin         # Ensures admin access
async def create_employee(
    employee_data: EmployeeCreate,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    # Route implementation
    pass
```

## Middleware Features

The system includes several middleware components:

### Request Tracking
- Assigns unique request IDs
- Logs request/response timing
- Adds tracking headers to responses

### Security Headers
- Adds security headers to all responses
- Configures HSTS for production
- Prevents clickjacking and XSS

### Rate Limiting
- Limits requests per IP address
- Configurable rates (default: 120 requests/minute)
- Returns `429 Too Many Requests` when exceeded

### Request Size Limiting
- Limits request body size (default: 50MB)
- Returns `413 Request Entity Too Large` when exceeded

## Error Logging

All errors are automatically logged with structured data:

```python
import logging

logger = logging.getLogger(__name__)

# Errors are logged with context
logger.error(
    "Database error in create_employee",
    extra={
        "request_id": "550e8400-e29b-41d4-a716-446655440000",
        "user_id": "user-123",
        "error_type": "DatabaseError",
        "original_error": str(original_exception)
    }
)
```

## Best Practices

### 1. Use Specific Exceptions
```python
# Good
raise ResourceNotFoundError("Employee", employee_id)

# Bad
raise HTTPException(status_code=404, detail="Not found")
```

### 2. Provide Context in Error Data
```python
# Good
raise ValidationError(
    "Invalid date range",
    error_data={
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "max_range_days": 365
    }
)

# Bad
raise ValidationError("Invalid date range")
```

### 3. Use Service Layer for Business Logic
```python
# Good - Service handles errors
employee_service = EmployeeService(db)
employee = employee_service.create_employee(employee_data)

# Bad - Route handles database directly
try:
    employee = Employee(**employee_data)
    db.add(employee)
    db.commit()
except IntegrityError:
    raise HTTPException(400, "Already exists")
```

### 4. Validate Input Early
```python
# Good - Validate at route entry
@router.post("/employees")
async def create_employee(employee_data: EmployeeCreate, ...):
    validate_employee_data(
        employee_data.name, 
        employee_data.email, 
        employee_data.department
    )
    # ... rest of route

# Bad - Let service handle all validation
```

### 5. Log Important Actions
```python
# Good - Log significant actions
self.log_service_action(
    "employee_created", 
    "Employee", 
    str(employee.id),
    extra_data={"created_by": str(current_user.id)}
)

# Bad - No audit trail
```

## Testing Error Handling

Test your error handling with example endpoints:

```bash
# Test validation error
curl -X GET "http://localhost:8000/api/v1/example/error-examples/validation"

# Test not found error
curl -X GET "http://localhost:8000/api/v1/example/error-examples/not-found"

# Test database error
curl -X GET "http://localhost:8000/api/v1/example/error-examples/database"

# Test file upload error
curl -X POST "http://localhost:8000/api/v1/example/error-examples/file-upload" \
     -F "file=@large_file.txt"
```

## Monitoring and Alerting

Error patterns to monitor:

1. **High error rates** (>5% of requests)
2. **Database connection errors**
3. **Authentication failures** (potential attacks)
4. **File upload errors** (storage issues)
5. **Face recognition errors** (model issues)

Set up alerts for:
- Error rate spikes
- Database connectivity issues
- Repeated authentication failures from same IP
- Service unavailability errors

## Migration Guide

To update existing routes to use the new error handling:

1. **Replace HTTPException with specific exceptions**:
   ```python
   # Old
   raise HTTPException(404, "Employee not found")
   
   # New
   raise ResourceNotFoundError("Employee", employee_id)
   ```

2. **Use service base class**:
   ```python
   # Old
   class MyService:
       def __init__(self, db: Session):
           self.db = db
   
   # New
   class MyService(BaseService):
       def __init__(self, db: Session):
           super().__init__(db)
   ```

3. **Add validation**:
   ```python
   # Old
   def create_employee(self, data):
       employee = Employee(**data)
       # ...
   
   # New
   def create_employee(self, data):
       self.validate_required_fields(data, ["name", "email"])
       validate_employee_data(data["name"], data["email"])
       # ...
   ```

4. **Add decorators to routes**:
   ```python
   # Old
   @router.post("/employees")
   async def create_employee(...):
   
   # New
   @router.post("/employees")
   @handle_service_errors
   @log_route_access
   async def create_employee(...):
   ```

This comprehensive error handling system provides better user experience, easier debugging, and more maintainable code.

