"""
Base Service Class with Enhanced Error Handling
"""

import logging
from typing import Optional, Any, Dict
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from app.core.exceptions import (
    DatabaseError,
    ResourceNotFoundError,
    ResourceAlreadyExistsError,
    ValidationError
)

logger = logging.getLogger(__name__)


class BaseService:
    """Base service class with common error handling patterns."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def safe_commit(self, error_message: str = "Database operation failed") -> bool:
        """Safely commit database transaction with error handling."""
        try:
            self.db.commit()
            return True
        except IntegrityError as e:
            self.db.rollback()
            logger.error(f"Integrity error during commit: {str(e)}")
            raise ResourceAlreadyExistsError(
                resource_type="Resource",
                error_data={"original_error": str(e)}
            )
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Database error during commit: {str(e)}")
            raise DatabaseError(
                detail=error_message,
                error_data={"original_error": str(e)}
            )
        except Exception as e:
            self.db.rollback()
            logger.error(f"Unexpected error during commit: {str(e)}")
            raise DatabaseError(
                detail=error_message,
                error_data={"original_error": str(e)}
            )
    
    def safe_rollback(self):
        """Safely rollback database transaction."""
        try:
            self.db.rollback()
        except Exception as e:
            logger.error(f"Error during rollback: {str(e)}")
    
    def get_or_404(self, model_class, resource_id: str, resource_type: str = None):
        """Get resource by ID or raise 404 error."""
        try:
            resource = self.db.query(model_class).filter(
                model_class.id == resource_id
            ).first()
            
            if not resource:
                raise ResourceNotFoundError(
                    resource_type=resource_type or model_class.__name__,
                    resource_id=resource_id
                )
            
            return resource
            
        except SQLAlchemyError as e:
            logger.error(f"Database error in get_or_404: {str(e)}")
            raise DatabaseError(
                detail=f"Error retrieving {resource_type or model_class.__name__}",
                error_data={"resource_id": resource_id, "original_error": str(e)}
            )
    
    def check_unique_constraint(
        self, 
        model_class, 
        field_name: str, 
        field_value: Any, 
        resource_type: str = None,
        exclude_id: str = None
    ):
        """Check if a field value is unique."""
        try:
            query = self.db.query(model_class).filter(
                getattr(model_class, field_name) == field_value
            )
            
            # Exclude current record if updating
            if exclude_id:
                query = query.filter(model_class.id != exclude_id)
            
            existing = query.first()
            
            if existing:
                raise ResourceAlreadyExistsError(
                    resource_type=resource_type or model_class.__name__,
                    field=field_name,
                    value=str(field_value)
                )
                
        except SQLAlchemyError as e:
            logger.error(f"Database error in unique constraint check: {str(e)}")
            raise DatabaseError(
                detail=f"Error checking uniqueness for {field_name}",
                error_data={"field": field_name, "value": field_value, "original_error": str(e)}
            )
    
    def validate_required_fields(self, data: Dict[str, Any], required_fields: list):
        """Validate that required fields are present and not empty."""
        missing_fields = []
        empty_fields = []
        
        for field in required_fields:
            if field not in data:
                missing_fields.append(field)
            elif data[field] is None or (isinstance(data[field], str) and not data[field].strip()):
                empty_fields.append(field)
        
        if missing_fields or empty_fields:
            error_details = {}
            if missing_fields:
                error_details["missing_fields"] = missing_fields
            if empty_fields:
                error_details["empty_fields"] = empty_fields
            
            raise ValidationError(
                detail="Required fields validation failed",
                error_data=error_details
            )
    
    def paginate_query(self, query, skip: int = 0, limit: int = 100):
        """Apply pagination to query with validation."""
        # Validate pagination parameters
        if skip < 0:
            raise ValidationError(
                detail="Skip parameter cannot be negative",
                field="skip",
                value=skip
            )
        
        if limit <= 0 or limit > 1000:
            raise ValidationError(
                detail="Limit parameter must be between 1 and 1000",
                field="limit",
                value=limit
            )
        
        return query.offset(skip).limit(limit)
    
    def log_service_action(
        self, 
        action: str, 
        resource_type: str = None, 
        resource_id: str = None, 
        extra_data: Dict[str, Any] = None
    ):
        """Log service actions for auditing."""
        log_data = {
            "action": action,
            "service": self.__class__.__name__
        }
        
        if resource_type:
            log_data["resource_type"] = resource_type
        if resource_id:
            log_data["resource_id"] = resource_id
        if extra_data:
            log_data.update(extra_data)
        
        logger.info(f"Service action: {action}", extra=log_data)
    
    def handle_file_validation(
        self, 
        file, 
        allowed_types: list = None, 
        max_size: int = None
    ):
        """Validate uploaded files."""
        from app.core.exceptions import InvalidFileError, FileTooLargeError
        
        if not file:
            raise InvalidFileError("No file provided")
        
        # Check file type
        if allowed_types and file.content_type:
            if not any(file.content_type.startswith(allowed_type) for allowed_type in allowed_types):
                raise InvalidFileError(
                    detail=f"Invalid file type. Allowed types: {', '.join(allowed_types)}",
                    file_type=file.content_type,
                    error_data={"allowed_types": allowed_types}
                )
        
        # Check file size (if we can determine it)
        if max_size and hasattr(file, 'size') and file.size:
            if file.size > max_size:
                raise FileTooLargeError(
                    max_size=max_size,
                    actual_size=file.size
                )
        
        return True

