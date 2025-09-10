from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import os
from app.core.config import settings
from app.core.database import engine, Base
from app.auth.routes import router as auth_router
from app.employees.routes import router as employees_router
from app.attendance.routes import router as attendance_router
from app.payrolls.routes import router as payrolls_router

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

# Mount static files for uploaded images
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Include routers
app.include_router(auth_router, prefix="/api/v1")
app.include_router(employees_router, prefix="/api/v1")
app.include_router(attendance_router, prefix="/api/v1")
app.include_router(payrolls_router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup."""
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        print("Database tables created successfully")
    except Exception as e:
        print(f"Error creating database tables: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("Application shutting down...")


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
            "payrolls": "/api/v1/payrolls"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "API is running"}


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "message": str(exc) if settings.debug else "An unexpected error occurred"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
