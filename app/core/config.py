from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # Database Configuration
    database_url: str = "postgresql://postgres:yaadrolens%401234@db.phvkwuirfumruwxnpxpy.supabase.co:5432/postgres"
    
    # JWT Configuration
    secret_key: str = "your-super-secret-jwt-key-change-this-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # Application Configuration
    app_name: str = "Face Recognition Attendance System"
    debug: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Face Recognition Configuration
    recognition_threshold: float = 0.5  # Balanced threshold for good accuracy and recognition
    embedding_model: str = "ArcFace"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Create settings instance
settings = Settings()
