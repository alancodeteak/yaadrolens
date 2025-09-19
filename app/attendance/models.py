from sqlalchemy import Column, String, DateTime, ForeignKey, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import enum
from app.core.database import Base, generate_uuid


class AttendanceType(str, enum.Enum):
    IN = "IN"
    OUT = "OUT"


class AttendanceLog(Base):
    __tablename__ = "attendance"

    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    employee_id = Column(UUID(as_uuid=True), ForeignKey("employees.id"), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    type = Column(Enum(AttendanceType), nullable=False)
    confidence_score = Column(String(10))  # Store the confidence score from face recognition
    image_path = Column(String(500))  # Path to the captured image
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    employee = relationship("Employee", back_populates="attendance_logs")
    ground_truth_validation = relationship("GroundTruthValidation", back_populates="attendance_log", uselist=False)
