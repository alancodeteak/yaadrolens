"""
Middleware for the Face Recognition Attendance System
"""

import time
import uuid
import logging
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from fastapi import status
from fastapi.responses import JSONResponse

from app.core.config import settings

logger = logging.getLogger(__name__)


class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware to track requests with unique IDs and timing."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Record start time
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "client_ip": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent"),
            }
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(round(process_time, 4))
            
            # Log response
            logger.info(
                f"Request completed: {request.method} {request.url.path} - {response.status_code}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "process_time": process_time,
                }
            )
            
            return response
            
        except Exception as exc:
            # Calculate processing time for failed requests
            process_time = time.time() - start_time
            
            # Log error
            logger.error(
                f"Request failed: {request.method} {request.url.path} - {type(exc).__name__}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "exception": str(exc),
                    "process_time": process_time,
                }
            )
            
            # Re-raise the exception to be handled by error handlers
            raise exc


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Add HSTS header for production
        if not settings.debug:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts = {}
        self.window_start_time = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health check and static files
        if request.url.path in ["/health", "/"] or request.url.path.startswith("/static"):
            return await call_next(request)
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Initialize tracking for new IPs
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = 0
            self.window_start_time[client_ip] = current_time
        
        # Reset counter if window has passed (1 minute)
        if current_time - self.window_start_time[client_ip] >= 60:
            self.request_counts[client_ip] = 0
            self.window_start_time[client_ip] = current_time
        
        # Check rate limit
        if self.request_counts[client_ip] >= self.requests_per_minute:
            logger.warning(
                f"Rate limit exceeded for IP: {client_ip}",
                extra={
                    "client_ip": client_ip,
                    "request_count": self.request_counts[client_ip],
                    "path": request.url.path,
                    "method": request.method
                }
            )
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": True,
                    "status_code": 429,
                    "detail": "Rate limit exceeded. Please try again later.",
                    "error_code": "RATE_LIMIT_EXCEEDED"
                },
                headers={"Retry-After": "60"}
            )
        
        # Increment request count
        self.request_counts[client_ip] += 1
        
        return await call_next(request)


class RequestSizeMiddleware(BaseHTTPMiddleware):
    """Middleware to limit request body size."""
    
    def __init__(self, app, max_request_size: int = 50 * 1024 * 1024):  # 50MB default
        super().__init__(app)
        self.max_request_size = max_request_size
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check content length
        content_length = request.headers.get("content-length")
        
        if content_length:
            content_length = int(content_length)
            if content_length > self.max_request_size:
                logger.warning(
                    f"Request body too large: {content_length} bytes (max: {self.max_request_size})",
                    extra={
                        "content_length": content_length,
                        "max_size": self.max_request_size,
                        "path": request.url.path,
                        "method": request.method,
                        "client_ip": request.client.host if request.client else None
                    }
                )
                
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={
                        "error": True,
                        "status_code": 413,
                        "detail": f"Request body too large. Maximum size: {self.max_request_size} bytes",
                        "error_code": "REQUEST_TOO_LARGE",
                        "error_data": {
                            "max_size": self.max_request_size,
                            "actual_size": content_length
                        }
                    }
                )
        
        return await call_next(request)


class DatabaseHealthMiddleware(BaseHTTPMiddleware):
    """Middleware to check database health for critical endpoints."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip health check for non-critical endpoints
        if request.url.path in ["/health", "/", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Skip for static files
        if request.url.path.startswith("/static") or request.url.path.startswith("/uploads"):
            return await call_next(request)
        
        # For API endpoints, we could add database health check here
        # For now, just pass through
        return await call_next(request)


def add_middleware(app):
    """Add all middleware to the FastAPI app."""
    
    # Add middleware in reverse order (last added is executed first)
    
    # Request size limiting
    app.add_middleware(RequestSizeMiddleware, max_request_size=50 * 1024 * 1024)  # 50MB
    
    # Rate limiting (only in production or if explicitly enabled)
    if not settings.debug or getattr(settings, 'enable_rate_limiting', False):
        app.add_middleware(RateLimitingMiddleware, requests_per_minute=120)
    
    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Request tracking (should be last to track everything)
    app.add_middleware(RequestTrackingMiddleware)
    
    logger.info("Middleware registered successfully")

