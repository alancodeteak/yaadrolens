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
    high_confidence_threshold: float = 0.8  # High confidence threshold for immediate recognition
    low_confidence_threshold: float = 0.3  # Low confidence threshold for retry attempts
    max_retry_attempts: int = 3  # Maximum retry attempts for low confidence
    embedding_model: str = "ArcFace"
    
    # Smart Recognition Features
    enable_confidence_scoring: bool = True
    enable_retry_mechanism: bool = True
    enable_quality_validation: bool = True
    enable_anti_spoofing: bool = True
    
    # Training Configuration
    training_photos_required: int = 15  # Number of training photos required per employee
    training_quality_threshold: float = 0.7  # Minimum quality score for training photos
    training_photo_max_size: int = 5 * 1024 * 1024  # 5MB max file size
    training_photo_formats: list = ["image/jpeg", "image/png", "image/jpg"]
    
    # Training Photo Requirements
    training_poses: list = ["frontal", "left", "right", "up", "down"]
    training_lighting: list = ["natural", "office", "soft", "low"]
    training_expressions: list = ["neutral", "smile", "serious"]
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379/0"
    redis_password: Optional[str] = None
    redis_max_connections: int = 20
    redis_socket_timeout: int = 5
    redis_socket_connect_timeout: int = 5
    
    # Cache Settings
    cache_ttl: int = 3600  # 1 hour
    bulk_cache_ttl: int = 1800  # 30 minutes
    enable_redis_cache: bool = True
    cache_warm_on_startup: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Create settings instance
settings = Settings()
