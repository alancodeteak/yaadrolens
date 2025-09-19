from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, Text, Integer, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base, generate_uuid


class Employee(Base):
    __tablename__ = "employees"

    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    department = Column(String(100))
    position = Column(String(100))
    phone = Column(String(20))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Training data fields
    training_photos_count = Column(Integer, default=0)
    training_quality_score = Column(Float, default=0.0)
    training_status = Column(String(20), default="pending")  # pending, collecting, completed
    last_training_photo = Column(DateTime(timezone=True))

    # Relationships
    embeddings = relationship("EmployeeEmbedding", back_populates="employee", cascade="all, delete-orphan")
    attendance_logs = relationship("AttendanceLog", back_populates="employee")
    payrolls = relationship("Payroll", back_populates="employee")
    training_photos = relationship("TrainingPhoto", back_populates="employee", cascade="all, delete-orphan")


class EmployeeEmbedding(Base):
    __tablename__ = "employee_embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    employee_id = Column(UUID(as_uuid=True), ForeignKey("employees.id"), nullable=False)
    embedding = Column(Text, nullable=False)  # JSON string of the embedding vector
    image_path = Column(String(500))  # Path to the original image
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    employee = relationship("Employee", back_populates="embeddings")


class TrainingPhoto(Base):
    __tablename__ = "training_photos"

    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    employee_id = Column(UUID(as_uuid=True), ForeignKey("employees.id"), nullable=False)
    image_path = Column(String(500), nullable=False)
    pose_type = Column(String(20))  # frontal, left, right, up, down
    quality_score = Column(Float)
    lighting_condition = Column(String(20))  # natural, office, soft, low
    expression = Column(String(20))  # neutral, smile, serious
    is_validated = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    employee = relationship("Employee", back_populates="training_photos")
