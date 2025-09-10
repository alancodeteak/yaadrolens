#!/usr/bin/env python3
"""
Simple script to run the Face Recognition Attendance System
"""

import uvicorn
from app.main import app
from app.core.config import settings

if __name__ == "__main__":
    print("🚀 Starting Face Recognition Attendance System...")
    print(f"📱 App: {settings.app_name}")
    print(f"🌐 Host: {settings.host}")
    print(f"🔌 Port: {settings.port}")
    print(f"🔧 Debug: {settings.debug}")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("-" * 50)
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )
