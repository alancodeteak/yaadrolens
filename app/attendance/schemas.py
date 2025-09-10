from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from uuid import UUID
from app.attendance.models import AttendanceType


class AttendanceLogCreate(BaseModel):
    employee_id: str
    type: AttendanceType


class AttendanceLogResponse(BaseModel):
    id: UUID
    employee_id: UUID
    timestamp: datetime
    type: AttendanceType
    confidence_score: Optional[str]
    image_path: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class AttendanceWithEmployee(AttendanceLogResponse):
    employee_name: str
    employee_email: str
    employee_department: Optional[str]


class ClockInOutRequest(BaseModel):
    # Note: Image will be handled as multipart form data
    pass


class ClockInOutResponse(BaseModel):
    success: bool
    message: str
    employee_id: Optional[str]
    employee_name: Optional[str]
    attendance_type: Optional[AttendanceType]
    confidence_score: Optional[float]
    timestamp: Optional[datetime]


class AttendanceReportRequest(BaseModel):
    start_date: datetime
    end_date: datetime
    employee_id: Optional[str] = None


class AttendanceReportResponse(BaseModel):
    employee_id: str
    employee_name: str
    employee_email: str
    total_days: int
    total_hours: float
    attendance_logs: List[AttendanceWithEmployee]
