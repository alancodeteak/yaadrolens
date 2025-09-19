from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os
import logging
from app.core.config import settings
from app.core.database import engine, Base
from app.core.redis_service import redis_service
from app.core.error_handlers import register_error_handlers
from app.core.middleware import add_middleware
from app.face_recognition.cached_recognition_service import CachedFaceRecognitionService
from app.auth.routes import router as auth_router
from app.employees.routes import router as employees_router
from app.attendance.routes import router as attendance_router
from app.payrolls.routes import router as payrolls_router
from app.ml_monitoring.routes import router as ml_monitoring_router

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="A production-grade Face Recognition Attendance System using FastAPI, DeepFace, and Supabase",
    version="1.0.0",
    debug=settings.debug
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directories
os.makedirs("uploads/employee_images", exist_ok=True)
os.makedirs("uploads/attendance_images", exist_ok=True)
os.makedirs("uploads/training_photos", exist_ok=True)

# Mount static files for uploaded images
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Mount static files for UI
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Register error handlers
register_error_handlers(app)

# Add middleware
add_middleware(app)

# Include routers
app.include_router(auth_router, prefix="/api/v1")
app.include_router(employees_router, prefix="/api/v1")
app.include_router(attendance_router, prefix="/api/v1")
app.include_router(payrolls_router, prefix="/api/v1")
app.include_router(ml_monitoring_router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    """Initialize database tables and warm cache on startup."""
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
        # Warm cache if enabled and Redis is available
        if settings.cache_warm_on_startup and settings.enable_redis_cache:
            try:
                if redis_service.is_available():
                    from app.core.database import SessionLocal
                    db = SessionLocal()
                    try:
                        cached_service = CachedFaceRecognitionService(redis_service, db)
                        success = cached_service.warm_cache()
                        if success:
                            logger.info("Face recognition cache warmed successfully")
                        else:
                            logger.warning("Failed to warm cache on startup")
                    finally:
                        db.close()
                else:
                    logger.warning("Redis not available, cache warming skipped")
            except Exception as cache_error:
                logger.error(f"Cache warming failed on startup: {cache_error}")
                
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise  # Re-raise to prevent app from starting with errors


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Application shutting down...")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Face Recognition Attendance System API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "auth": "/api/v1/auth",
            "employees": "/api/v1/employees",
            "attendance": "/api/v1/attendance",
            "payrolls": "/api/v1/payrolls",
            "ml_monitoring": "/api/v1/ml-monitoring",
            "training": "/api/v1/employees/{employee_id}/training-*",
            "smart_attendance": "/api/v1/attendance/smart-clock",
            "recognition_stats": "/api/v1/attendance/recognition-stats"
        },
        "training_ui": "/static/training_collection.html"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "API is running"}


# Note: Global exception handlers are now registered in error_handlers.py


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
