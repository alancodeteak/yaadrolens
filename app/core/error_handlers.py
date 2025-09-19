"""
Global Error Handlers for the Face Recognition Attendance System
"""

import logging
import traceback
from typing import Dict, Any
from datetime import datetime

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError, HTTPException
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy.exc import (
    SQLAlchemyError, 
    IntegrityError, 
    DataError, 
    OperationalError,
    InvalidRequestError
)
from psycopg2.errors import (
    UniqueViolation, 
    ForeignKeyViolation, 
    InvalidTextRepresentation,
    ConnectionException
)

from app.core.exceptions import BaseAPIException
from app.core.config import settings

# Set up logger
logger = logging.getLogger(__name__)


def create_error_response(
    status_code: int,
    detail: str,
    error_code: str = None,
    error_data: Dict[str, Any] = None,
    request_id: str = None,
    timestamp: str = None
) -> JSONResponse:
    """Create standardized error response."""
    
    content = {
        "error": True,
        "status_code": status_code,
        "detail": detail,
        "timestamp": timestamp or datetime.utcnow().isoformat(),
    }
    
    if error_code:
        content["error_code"] = error_code
    
    if error_data:
        content["error_data"] = error_data
    
    if request_id:
        content["request_id"] = request_id
    
    # Add debug info in development
    if settings.debug and error_data:
        content["debug_info"] = error_data
    
    return JSONResponse(
        status_code=status_code,
        content=content
    )


async def base_api_exception_handler(request: Request, exc: BaseAPIException) -> JSONResponse:
    """Handle custom API exceptions."""
    
    request_id = getattr(request.state, 'request_id', None)
    
    logger.warning(
        f"API Exception: {exc.error_code or 'UNKNOWN'} - {exc.detail}",
        extra={
            "status_code": exc.status_code,
            "error_code": exc.error_code,
            "error_data": exc.error_data,
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method
        }
    )
    
    return create_error_response(
        status_code=exc.status_code,
        detail=exc.detail,
        error_code=exc.error_code,
        error_data=exc.error_data,
        request_id=request_id
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle FastAPI HTTP exceptions."""
    
    request_id = getattr(request.state, 'request_id', None)
    
    logger.warning(
        f"HTTP Exception: {exc.status_code} - {exc.detail}",
        extra={
            "status_code": exc.status_code,
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method
        }
    )
    
    return create_error_response(
        status_code=exc.status_code,
        detail=exc.detail,
        error_code="HTTP_EXCEPTION",
        request_id=request_id
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle request validation errors."""
    
    request_id = getattr(request.state, 'request_id', None)
    
    # Extract validation errors
    validation_errors = []
    for error in exc.errors():
        validation_errors.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
            "input": error.get("input")
        })
    
    logger.warning(
        f"Validation Error: {len(validation_errors)} validation error(s)",
        extra={
            "validation_errors": validation_errors,
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method
        }
    )
    
    return create_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="Request validation failed",
        error_code="VALIDATION_ERROR",
        error_data={"validation_errors": validation_errors},
        request_id=request_id
    )


async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    """Handle SQLAlchemy database exceptions."""
    
    request_id = getattr(request.state, 'request_id', None)
    
    # Determine error type and create appropriate response
    if isinstance(exc, IntegrityError):
        # Handle specific database constraint violations
        if isinstance(exc.orig, UniqueViolation):
            detail = "Resource already exists with the provided data"
            error_code = "DUPLICATE_RESOURCE"
            status_code = status.HTTP_409_CONFLICT
        elif isinstance(exc.orig, ForeignKeyViolation):
            detail = "Referenced resource does not exist"
            error_code = "INVALID_REFERENCE"
            status_code = status.HTTP_400_BAD_REQUEST
        else:
            detail = "Data integrity constraint violated"
            error_code = "INTEGRITY_ERROR"
            status_code = status.HTTP_400_BAD_REQUEST
    
    elif isinstance(exc, DataError):
        if isinstance(exc.orig, InvalidTextRepresentation):
            detail = "Invalid data format provided"
            error_code = "INVALID_DATA_FORMAT"
        else:
            detail = "Invalid data provided"
            error_code = "DATA_ERROR"
        status_code = status.HTTP_400_BAD_REQUEST
    
    elif isinstance(exc, OperationalError):
        if isinstance(exc.orig, ConnectionException):
            detail = "Database connection failed"
            error_code = "DATABASE_CONNECTION_ERROR"
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        else:
            detail = "Database operation failed"
            error_code = "DATABASE_OPERATION_ERROR"
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    
    elif isinstance(exc, InvalidRequestError):
        detail = "Invalid database request"
        error_code = "INVALID_DB_REQUEST"
        status_code = status.HTTP_400_BAD_REQUEST
    
    else:
        detail = "Database error occurred"
        error_code = "DATABASE_ERROR"
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    
    # Log the error
    logger.error(
        f"Database Error: {error_code} - {detail}",
        extra={
            "exception_type": type(exc).__name__,
            "error_details": str(exc),
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method
        }
    )
    
    # Include original error details in debug mode
    error_data = {}
    if settings.debug:
        error_data = {
            "exception_type": type(exc).__name__,
            "original_error": str(exc)
        }
    
    return create_error_response(
        status_code=status_code,
        detail=detail,
        error_code=error_code,
        error_data=error_data,
        request_id=request_id
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle all other unhandled exceptions."""
    
    request_id = getattr(request.state, 'request_id', None)
    
    # Log the full exception with traceback
    logger.error(
        f"Unhandled Exception: {type(exc).__name__} - {str(exc)}",
        extra={
            "exception_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method
        }
    )
    
    # Create error response
    if settings.debug:
        # Include detailed error information in debug mode
        detail = f"Internal server error: {str(exc)}"
        error_data = {
            "exception_type": type(exc).__name__,
            "traceback": traceback.format_exc().split('\n')
        }
    else:
        # Generic error message in production
        detail = "An unexpected error occurred. Please try again later."
        error_data = None
    
    return create_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=detail,
        error_code="INTERNAL_SERVER_ERROR",
        error_data=error_data,
        request_id=request_id
    )


async def starlette_http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle Starlette HTTP exceptions."""
    
    request_id = getattr(request.state, 'request_id', None)
    
    logger.warning(
        f"Starlette HTTP Exception: {exc.status_code} - {exc.detail}",
        extra={
            "status_code": exc.status_code,
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method
        }
    )
    
    return create_error_response(
        status_code=exc.status_code,
        detail=exc.detail,
        error_code="HTTP_EXCEPTION",
        request_id=request_id
    )


# Error handler mapping
ERROR_HANDLERS = {
    BaseAPIException: base_api_exception_handler,
    HTTPException: http_exception_handler,
    StarletteHTTPException: starlette_http_exception_handler,
    RequestValidationError: validation_exception_handler,
    SQLAlchemyError: sqlalchemy_exception_handler,
    Exception: generic_exception_handler,
}


def register_error_handlers(app):
    """Register all error handlers with the FastAPI app."""
    
    for exception_class, handler in ERROR_HANDLERS.items():
        app.add_exception_handler(exception_class, handler)
    
    logger.info("Error handlers registered successfully")

