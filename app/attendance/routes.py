from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, date
from app.core.database import get_db
from app.core.dependencies import get_current_admin_user
from app.auth.models import User
from app.attendance.schemas import (
    AttendanceLogResponse,
    AttendanceWithEmployee,
    ClockInOutResponse,
    AttendanceReportRequest,
    AttendanceReportResponse
)
from app.attendance.service import AttendanceService
from app.face_recognition.face_quality_utils import face_quality_validator, face_quality_validator_strict
from app.face_recognition.smart_recognition_service import smart_recognition_service

router = APIRouter(prefix="/attendance", tags=["attendance"])


@router.post("/clock", response_model=ClockInOutResponse)
async def clock_in_out(
    image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Clock in/out using face recognition."""
    attendance_service = AttendanceService(db)
    
    success, message, employee_id, employee_name, attendance_type, confidence_score, timestamp = await attendance_service.clock_in_out(image)
    
    return ClockInOutResponse(
        success=success,
        message=message,
        employee_id=employee_id,
        employee_name=employee_name,
        attendance_type=attendance_type,
        confidence_score=confidence_score,
        timestamp=timestamp
    )


@router.get("/logs", response_model=List[AttendanceWithEmployee])
async def get_attendance_logs(
    start_date: datetime = Query(...),
    end_date: datetime = Query(...),
    employee_id: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get attendance logs for a date range."""
    attendance_service = AttendanceService(db)
    logs = attendance_service.get_attendance_logs(
        start_date=start_date,
        end_date=end_date,
        employee_id=employee_id,
        skip=skip,
        limit=limit
    )
    
    return [
        AttendanceWithEmployee(
            id=str(log.id),
            employee_id=str(log.employee_id),
            timestamp=log.timestamp,
            type=log.type,
            confidence_score=log.confidence_score,
            image_path=log.image_path,
            created_at=log.created_at,
            employee_name=log.employee.name,
            employee_email=log.employee.email,
            employee_department=log.employee.department
        )
        for log in logs
    ]


@router.get("/report/{employee_id}", response_model=AttendanceReportResponse)
async def get_employee_attendance_report(
    employee_id: str,
    start_date: datetime = Query(...),
    end_date: datetime = Query(...),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get attendance report for a specific employee."""
    attendance_service = AttendanceService(db)
    
    try:
        report = attendance_service.get_employee_attendance_report(
            employee_id=employee_id,
            start_date=start_date,
            end_date=end_date
        )
        
        return AttendanceReportResponse(
            employee_id=report["employee_id"],
            employee_name=report["employee_name"],
            employee_email=report["employee_email"],
            total_days=report["total_days"],
            total_hours=report["total_hours"],
            attendance_logs=[
                AttendanceWithEmployee(
                    id=str(log.id),
                    employee_id=str(log.employee_id),
                    timestamp=log.timestamp,
                    type=log.type,
                    confidence_score=log.confidence_score,
                    image_path=log.image_path,
                    created_at=log.created_at,
                    employee_name=log.employee.name,
                    employee_email=log.employee.email,
                    employee_department=log.employee.department
                )
                for log in report["attendance_logs"]
            ]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating report: {str(e)}"
        )


@router.get("/daily-summary", response_model=List[dict])
async def get_daily_attendance_summary(
    date: date = Query(default_factory=date.today),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get daily attendance summary for all employees."""
    attendance_service = AttendanceService(db)
    
    # Convert date to datetime
    target_date = datetime.combine(date, datetime.min.time())
    summary = attendance_service.get_daily_attendance_summary(target_date)
    
    return summary


@router.get("/employee/{employee_id}/today", response_model=List[AttendanceLogResponse])
async def get_employee_today_attendance(
    employee_id: str,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get today's attendance for a specific employee."""
    attendance_service = AttendanceService(db)
    
    today = datetime.now().date()
    start_of_day = datetime.combine(today, datetime.min.time())
    end_of_day = datetime.combine(today, datetime.max.time())
    
    logs = attendance_service.get_attendance_logs(
        start_date=start_of_day,
        end_date=end_of_day,
        employee_id=employee_id
    )
    
    return [
        AttendanceLogResponse(
            id=str(log.id),
            employee_id=str(log.employee_id),
            timestamp=log.timestamp,
            type=log.type,
            confidence_score=log.confidence_score,
            image_path=log.image_path,
            created_at=log.created_at
        )
        for log in logs
    ]


@router.get("/stats", response_model=dict)
async def get_attendance_stats(
    start_date: datetime = Query(...),
    end_date: datetime = Query(...),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get attendance statistics for a date range."""
    attendance_service = AttendanceService(db)
    
    # Get all logs for the period
    logs = attendance_service.get_attendance_logs(start_date, end_date)
    
    # Calculate statistics
    total_employees = len(set(log.employee_id for log in logs))
    total_clock_ins = len([log for log in logs if log.type.value == "IN"])
    total_clock_outs = len([log for log in logs if log.type.value == "OUT"])
    
    # Calculate average hours per employee
    employee_hours = {}
    for log in logs:
        employee_id = str(log.employee_id)
        if employee_id not in employee_hours:
            employee_hours[employee_id] = []
        
        if log.type.value == "IN":
            employee_hours[employee_id].append(("IN", log.timestamp))
        elif log.type.value == "OUT":
            employee_hours[employee_id].append(("OUT", log.timestamp))
    
    total_hours = 0
    for employee_id, events in employee_hours.items():
        events.sort(key=lambda x: x[1])  # Sort by timestamp
        clock_in_time = None
        for event_type, timestamp in events:
            if event_type == "IN":
                clock_in_time = timestamp
            elif event_type == "OUT" and clock_in_time:
                hours = (timestamp - clock_in_time).total_seconds() / 3600
                total_hours += hours
                clock_in_time = None
    
    avg_hours_per_employee = total_hours / total_employees if total_employees > 0 else 0
    
    return {
        "total_employees": total_employees,
        "total_clock_ins": total_clock_ins,
        "total_clock_outs": total_clock_outs,
        "total_hours_worked": round(total_hours, 2),
        "average_hours_per_employee": round(avg_hours_per_employee, 2),
        "period_start": start_date,
        "period_end": end_date
    }


@router.post("/check_face_quality")
async def check_face_quality(
    image: UploadFile = File(...),
    strict_mode: bool = False
):
    """Check face quality before allowing clock in/out."""
    import tempfile
    import os
    
    try:
        # Validate file type
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image file"
            )
        
        # Save temporary image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            content = await image.read()
            tmp_file.write(content)
            tmp_file.flush()
            
            try:
                # Validate face quality - use strict mode if requested
                validator = face_quality_validator_strict if strict_mode else face_quality_validator
                quality_result = validator.validate_face_quality(tmp_file.name)
                
                return {
                    "is_clear": bool(quality_result.is_clear),
                    "confidence": float(quality_result.confidence),
                    "issues": list(quality_result.issues),
                    "face_detected": bool(quality_result.face_detected),
                    "hands_detected": bool(quality_result.hands_detected),
                    "brightness_ok": bool(quality_result.brightness_ok),
                    "blur_ok": bool(quality_result.blur_ok),
                    "face_area_ratio": float(quality_result.face_area_ratio),
                    "message": "Face is clear and ready for recognition" if quality_result.is_clear else f"Face quality issues: {', '.join(quality_result.issues)}"
                }
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_file.name)
                except:
                    pass
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Face quality check error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking face quality: {str(e)}"
        )


@router.post("/smart-clock")
async def smart_clock_in_out(
    image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Smart clock in/out with enhanced recognition features."""
    attendance_service = AttendanceService(db)
    
    result = await attendance_service.smart_clock_in_out(image)
    return result


@router.get("/recognition-stats")
async def get_recognition_stats(
    current_user: User = Depends(get_current_admin_user)
):
    """Get face recognition performance statistics."""
    stats = smart_recognition_service.get_recognition_stats()
    return stats


@router.post("/reset-recognition-stats")
async def reset_recognition_stats(
    current_user: User = Depends(get_current_admin_user)
):
    """Reset face recognition performance statistics."""
    smart_recognition_service.reset_stats()
    return {"message": "Recognition statistics reset successfully"}
