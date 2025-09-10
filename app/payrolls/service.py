from typing import List, Optional, Dict
from datetime import datetime, date
from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import and_, func
from fastapi import HTTPException, status
from app.payrolls.models import Payroll
from app.employees.models import Employee
from app.attendance.models import AttendanceLog, AttendanceType
from app.attendance.service import AttendanceService


class PayrollService:
    def __init__(self, db: Session):
        self.db = db
        self.attendance_service = AttendanceService(db)
    
    def calculate_payroll(
        self, 
        employee_id: str, 
        month: int, 
        year: int, 
        hourly_rate: Decimal,
        overtime_rate: Optional[Decimal] = None,
        deductions: Decimal = Decimal('0')
    ) -> Dict:
        """Calculate payroll for an employee for a specific month."""
        try:
            # Get employee
            employee = self.db.query(Employee).filter(Employee.id == employee_id).first()
            if not employee:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Employee not found"
                )
            
            # Set default overtime rate (1.5x hourly rate)
            if overtime_rate is None:
                overtime_rate = hourly_rate * Decimal('1.5')
            
            # Get attendance logs for the month
            start_date = datetime(year, month, 1)
            if month == 12:
                end_date = datetime(year + 1, 1, 1) - datetime.timedelta(days=1)
            else:
                end_date = datetime(year, month + 1, 1) - datetime.timedelta(days=1)
            
            attendance_logs = self.attendance_service.get_attendance_logs(
                start_date=start_date,
                end_date=end_date,
                employee_id=employee_id
            )
            
            # Calculate hours worked
            hours_worked, overtime_hours = self._calculate_hours_worked(attendance_logs)
            
            # Calculate salary
            base_salary = hours_worked * hourly_rate
            overtime_pay = overtime_hours * overtime_rate
            gross_salary = base_salary + overtime_pay
            net_salary = gross_salary - deductions
            
            # Count attendance days
            attendance_days = len(set(log.timestamp.date() for log in attendance_logs))
            
            return {
                "employee_id": employee_id,
                "employee_name": employee.name,
                "month": month,
                "year": year,
                "hours_worked": hours_worked,
                "overtime_hours": overtime_hours,
                "hourly_rate": hourly_rate,
                "overtime_rate": overtime_rate,
                "base_salary": base_salary,
                "overtime_pay": overtime_pay,
                "gross_salary": gross_salary,
                "deductions": deductions,
                "net_salary": net_salary,
                "attendance_days": attendance_days
            }
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error calculating payroll: {str(e)}"
            )
    
    def _calculate_hours_worked(self, attendance_logs: List[AttendanceLog]) -> tuple[Decimal, Decimal]:
        """Calculate regular and overtime hours from attendance logs."""
        # Group logs by date
        daily_logs = {}
        for log in attendance_logs:
            date = log.timestamp.date()
            if date not in daily_logs:
                daily_logs[date] = []
            daily_logs[date].append(log)
        
        total_hours = Decimal('0')
        overtime_hours = Decimal('0')
        regular_hours_per_day = Decimal('8')  # Standard 8-hour workday
        
        for date, logs in daily_logs.items():
            # Sort logs by timestamp
            logs.sort(key=lambda x: x.timestamp)
            
            # Calculate hours for this day
            clock_in_time = None
            daily_hours = Decimal('0')
            
            for log in logs:
                if log.type == AttendanceType.IN and clock_in_time is None:
                    clock_in_time = log.timestamp
                elif log.type == AttendanceType.OUT and clock_in_time is not None:
                    # Calculate hours worked
                    hours_worked = Decimal(str((log.timestamp - clock_in_time).total_seconds() / 3600))
                    daily_hours += hours_worked
                    clock_in_time = None
            
            total_hours += daily_hours
            
            # Calculate overtime (hours over 8 per day)
            if daily_hours > regular_hours_per_day:
                overtime_hours += daily_hours - regular_hours_per_day
        
        return total_hours, overtime_hours
    
    def create_payroll_record(self, payroll_data: Dict) -> Payroll:
        """Create a payroll record in the database."""
        try:
            # Check if payroll already exists for this employee and month
            existing_payroll = self.db.query(Payroll).filter(
                and_(
                    Payroll.employee_id == payroll_data["employee_id"],
                    Payroll.month == payroll_data["month"],
                    Payroll.year == payroll_data["year"]
                )
            ).first()
            
            if existing_payroll:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Payroll already exists for this employee and month"
                )
            
            # Create payroll record
            payroll = Payroll(
                employee_id=payroll_data["employee_id"],
                month=payroll_data["month"],
                year=payroll_data["year"],
                salary=payroll_data["gross_salary"],
                hours_worked=payroll_data["hours_worked"],
                hourly_rate=payroll_data["hourly_rate"],
                overtime_hours=payroll_data["overtime_hours"],
                overtime_rate=payroll_data["overtime_rate"],
                deductions=payroll_data["deductions"],
                net_salary=payroll_data["net_salary"],
                status="pending"
            )
            
            self.db.add(payroll)
            self.db.commit()
            self.db.refresh(payroll)
            
            return payroll
            
        except HTTPException:
            self.db.rollback()
            raise
        except Exception as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error creating payroll record: {str(e)}"
            )
    
    def get_payroll(self, payroll_id: str) -> Optional[Payroll]:
        """Get payroll by ID."""
        return self.db.query(Payroll).filter(Payroll.id == payroll_id).first()
    
    def get_employee_payrolls(
        self, 
        employee_id: str, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[Payroll]:
        """Get all payrolls for an employee."""
        return self.db.query(Payroll).filter(
            Payroll.employee_id == employee_id
        ).order_by(Payroll.year.desc(), Payroll.month.desc()).offset(skip).limit(limit).all()
    
    def get_payrolls_by_period(
        self, 
        start_month: int, 
        start_year: int, 
        end_month: int, 
        end_year: int,
        employee_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Payroll]:
        """Get payrolls for a specific period."""
        query = self.db.query(Payroll)
        
        # Filter by period
        if start_year == end_year:
            query = query.filter(
                and_(
                    Payroll.year == start_year,
                    Payroll.month >= start_month,
                    Payroll.month <= end_month
                )
            )
        else:
            query = query.filter(
                and_(
                    (Payroll.year > start_year) | 
                    (Payroll.year == start_year and Payroll.month >= start_month),
                    (Payroll.year < end_year) | 
                    (Payroll.year == end_year and Payroll.month <= end_month)
                )
            )
        
        if employee_id:
            query = query.filter(Payroll.employee_id == employee_id)
        
        return query.order_by(Payroll.year.desc(), Payroll.month.desc()).offset(skip).limit(limit).all()
    
    def update_payroll(self, payroll_id: str, update_data: Dict) -> Optional[Payroll]:
        """Update payroll record."""
        try:
            payroll = self.db.query(Payroll).filter(Payroll.id == payroll_id).first()
            
            if not payroll:
                return None
            
            # Update fields
            for field, value in update_data.items():
                if hasattr(payroll, field):
                    setattr(payroll, field, value)
            
            self.db.commit()
            self.db.refresh(payroll)
            
            return payroll
            
        except Exception as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error updating payroll: {str(e)}"
            )
    
    def approve_payroll(self, payroll_id: str) -> bool:
        """Approve a payroll record."""
        try:
            payroll = self.db.query(Payroll).filter(Payroll.id == payroll_id).first()
            
            if not payroll:
                return False
            
            payroll.status = "approved"
            self.db.commit()
            
            return True
            
        except Exception as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error approving payroll: {str(e)}"
            )
    
    def mark_payroll_paid(self, payroll_id: str) -> bool:
        """Mark a payroll as paid."""
        try:
            payroll = self.db.query(Payroll).filter(Payroll.id == payroll_id).first()
            
            if not payroll:
                return False
            
            payroll.status = "paid"
            self.db.commit()
            
            return True
            
        except Exception as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error marking payroll as paid: {str(e)}"
            )
