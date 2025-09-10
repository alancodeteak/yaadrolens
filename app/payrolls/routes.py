from fastapi import APIRouter, Depends, HTTPException, status, Query, Response
from sqlalchemy.orm import Session
from typing import List, Optional
from decimal import Decimal
import csv
import io
from app.core.database import get_db
from app.core.dependencies import get_current_admin_user
from app.auth.models import User
from app.payrolls.schemas import (
    PayrollCreate,
    PayrollUpdate,
    PayrollResponse,
    PayrollWithEmployee,
    PayrollCalculationRequest,
    PayrollCalculationResponse,
    PayrollReportRequest
)
from app.payrolls.service import PayrollService

router = APIRouter(prefix="/payrolls", tags=["payrolls"])


@router.post("/calculate", response_model=PayrollCalculationResponse)
async def calculate_payroll(
    calculation_request: PayrollCalculationRequest,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Calculate payroll for an employee without creating a record."""
    payroll_service = PayrollService(db)
    
    try:
        result = payroll_service.calculate_payroll(
            employee_id=calculation_request.employee_id,
            month=calculation_request.month,
            year=calculation_request.year,
            hourly_rate=calculation_request.hourly_rate,
            overtime_rate=calculation_request.overtime_rate,
            deductions=calculation_request.deductions
        )
        
        return PayrollCalculationResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculating payroll: {str(e)}"
        )


@router.post("/", response_model=PayrollResponse, status_code=status.HTTP_201_CREATED)
async def create_payroll(
    payroll_data: PayrollCreate,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Create a payroll record."""
    payroll_service = PayrollService(db)
    
    try:
        # First calculate the payroll
        calculation_result = payroll_service.calculate_payroll(
            employee_id=payroll_data.employee_id,
            month=payroll_data.month,
            year=payroll_data.year,
            hourly_rate=payroll_data.hourly_rate or Decimal('15.00'),  # Default hourly rate
            overtime_rate=payroll_data.overtime_rate,
            deductions=payroll_data.deductions
        )
        
        # Create the payroll record
        payroll = payroll_service.create_payroll_record(calculation_result)
        
        return payroll
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating payroll: {str(e)}"
        )


@router.get("/", response_model=List[PayrollWithEmployee])
async def get_payrolls(
    start_month: int = Query(...),
    start_year: int = Query(...),
    end_month: int = Query(...),
    end_year: int = Query(...),
    employee_id: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get payrolls for a specific period."""
    payroll_service = PayrollService(db)
    
    payrolls = payroll_service.get_payrolls_by_period(
        start_month=start_month,
        start_year=start_year,
        end_month=end_month,
        end_year=end_year,
        employee_id=employee_id,
        skip=skip,
        limit=limit
    )
    
    return [
        PayrollWithEmployee(
            id=str(payroll.id),
            employee_id=str(payroll.employee_id),
            month=payroll.month,
            year=payroll.year,
            salary=payroll.salary,
            hours_worked=payroll.hours_worked,
            hourly_rate=payroll.hourly_rate,
            overtime_hours=payroll.overtime_hours,
            overtime_rate=payroll.overtime_rate,
            deductions=payroll.deductions,
            net_salary=payroll.net_salary,
            status=payroll.status,
            processed_at=payroll.processed_at,
            created_at=payroll.created_at,
            employee_name=payroll.employee.name,
            employee_email=payroll.employee.email,
            employee_department=payroll.employee.department
        )
        for payroll in payrolls
    ]


@router.get("/{payroll_id}", response_model=PayrollWithEmployee)
async def get_payroll(
    payroll_id: str,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get payroll by ID."""
    payroll_service = PayrollService(db)
    payroll = payroll_service.get_payroll(payroll_id)
    
    if not payroll:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Payroll not found"
        )
    
    return PayrollWithEmployee(
        id=str(payroll.id),
        employee_id=str(payroll.employee_id),
        month=payroll.month,
        year=payroll.year,
        salary=payroll.salary,
        hours_worked=payroll.hours_worked,
        hourly_rate=payroll.hourly_rate,
        overtime_hours=payroll.overtime_hours,
        overtime_rate=payroll.overtime_rate,
        deductions=payroll.deductions,
        net_salary=payroll.net_salary,
        status=payroll.status,
        processed_at=payroll.processed_at,
        created_at=payroll.created_at,
        employee_name=payroll.employee.name,
        employee_email=payroll.employee.email,
        employee_department=payroll.employee.department
    )


@router.get("/employee/{employee_id}", response_model=List[PayrollResponse])
async def get_employee_payrolls(
    employee_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Get all payrolls for a specific employee."""
    payroll_service = PayrollService(db)
    payrolls = payroll_service.get_employee_payrolls(
        employee_id=employee_id,
        skip=skip,
        limit=limit
    )
    
    return payrolls


@router.put("/{payroll_id}", response_model=PayrollResponse)
async def update_payroll(
    payroll_id: str,
    update_data: PayrollUpdate,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Update payroll record."""
    payroll_service = PayrollService(db)
    
    update_dict = update_data.dict(exclude_unset=True)
    payroll = payroll_service.update_payroll(payroll_id, update_dict)
    
    if not payroll:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Payroll not found"
        )
    
    return payroll


@router.patch("/{payroll_id}/approve", response_model=PayrollResponse)
async def approve_payroll(
    payroll_id: str,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Approve a payroll record."""
    payroll_service = PayrollService(db)
    
    success = payroll_service.approve_payroll(payroll_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Payroll not found"
        )
    
    payroll = payroll_service.get_payroll(payroll_id)
    return payroll


@router.patch("/{payroll_id}/mark-paid", response_model=PayrollResponse)
async def mark_payroll_paid(
    payroll_id: str,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Mark a payroll as paid."""
    payroll_service = PayrollService(db)
    
    success = payroll_service.mark_payroll_paid(payroll_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Payroll not found"
        )
    
    payroll = payroll_service.get_payroll(payroll_id)
    return payroll


@router.get("/export/csv")
async def export_payrolls_csv(
    start_month: int = Query(...),
    start_year: int = Query(...),
    end_month: int = Query(...),
    end_year: int = Query(...),
    employee_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Export payrolls to CSV file."""
    payroll_service = PayrollService(db)
    
    payrolls = payroll_service.get_payrolls_by_period(
        start_month=start_month,
        start_year=start_year,
        end_month=end_month,
        end_year=end_year,
        employee_id=employee_id
    )
    
    # Create CSV content
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow([
        'Employee ID', 'Employee Name', 'Employee Email', 'Department',
        'Month', 'Year', 'Hours Worked', 'Overtime Hours', 'Hourly Rate',
        'Overtime Rate', 'Base Salary', 'Overtime Pay', 'Gross Salary',
        'Deductions', 'Net Salary', 'Status', 'Processed At'
    ])
    
    # Write data
    for payroll in payrolls:
        base_salary = payroll.hours_worked * payroll.hourly_rate
        overtime_pay = payroll.overtime_hours * payroll.overtime_rate
        gross_salary = base_salary + overtime_pay
        
        writer.writerow([
            str(payroll.employee_id),
            payroll.employee.name,
            payroll.employee.email,
            payroll.employee.department or '',
            payroll.month,
            payroll.year,
            float(payroll.hours_worked),
            float(payroll.overtime_hours),
            float(payroll.hourly_rate),
            float(payroll.overtime_rate),
            float(base_salary),
            float(overtime_pay),
            float(payroll.salary),
            float(payroll.deductions),
            float(payroll.net_salary),
            payroll.status,
            payroll.processed_at.isoformat()
        ])
    
    # Prepare response
    output.seek(0)
    csv_content = output.getvalue()
    output.close()
    
    filename = f"payrolls_{start_year}_{start_month:02d}_to_{end_year}_{end_month:02d}.csv"
    
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
