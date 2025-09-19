"""
Route Decorators for Enhanced Error Handling
"""

import functools
import logging
from typing import Callable, Any
from fastapi import Request

from app.core.exceptions import (
    ValidationError,
    DatabaseError,
    ExternalServiceError
)

logger = logging.getLogger(__name__)


def handle_service_errors(func: Callable) -> Callable:
    """Decorator to handle common service errors in route handlers."""
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        try:
            return await func(*args, **kwargs)
        except ValidationError:
            # Let validation errors pass through
            raise
        except DatabaseError:
            # Let database errors pass through
            raise
        except ExternalServiceError:
            # Let external service errors pass through
            raise
        except Exception as e:
            # Log unexpected errors
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            # Re-raise to be handled by global error handler
            raise
    
    return wrapper


def log_route_access(func: Callable) -> Callable:
    """Decorator to log route access for auditing."""
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        # Try to extract request and user info
        request = None
        current_user = None
        
        for arg in args:
            if isinstance(arg, Request):
                request = arg
                break
        
        # Look for current_user in kwargs
        if 'current_user' in kwargs:
            current_user = kwargs['current_user']
        
        # Log access
        log_data = {
            "function": func.__name__,
            "module": func.__module__
        }
        
        if request:
            log_data.update({
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else None
            })
        
        if current_user:
            log_data.update({
                "user_id": str(current_user.id),
                "user_email": current_user.email,
                "user_role": current_user.role
            })
        
        logger.info(f"Route access: {func.__name__}", extra=log_data)
        
        return await func(*args, **kwargs)
    
    return wrapper


def validate_content_type(allowed_types: list):
    """Decorator to validate request content type."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Find request object
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if request and request.headers.get("content-type"):
                content_type = request.headers.get("content-type").split(';')[0]
                if content_type not in allowed_types:
                    raise ValidationError(
                        detail=f"Invalid content type. Allowed: {', '.join(allowed_types)}",
                        field="content-type",
                        value=content_type,
                        error_data={"allowed_types": allowed_types}
                    )
            
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def require_admin(func: Callable) -> Callable:
    """Decorator to ensure only admin users can access the endpoint."""
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        # Look for current_user in kwargs
        current_user = kwargs.get('current_user')
        
        if not current_user:
            logger.warning(f"Admin access attempted without user context: {func.__name__}")
            raise ValidationError("User context required")
        
        if current_user.role not in ['admin', 'hr']:
            logger.warning(
                f"Non-admin user attempted admin access: {current_user.email} -> {func.__name__}",
                extra={
                    "user_id": str(current_user.id),
                    "user_role": current_user.role,
                    "function": func.__name__
                }
            )
            from app.core.exceptions import InsufficientPermissionsError
            raise InsufficientPermissionsError(
                detail=f"Admin access required for {func.__name__}",
                error_data={
                    "required_roles": ["admin", "hr"],
                    "user_role": current_user.role
                }
            )
        
        return await func(*args, **kwargs)
    
    return wrapper

