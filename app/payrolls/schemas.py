from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from decimal import Decimal


class PayrollCreate(BaseModel):
    employee_id: str
    month: int
    year: int
    hourly_rate: Optional[Decimal] = None
    overtime_rate: Optional[Decimal] = None
    deductions: Optional[Decimal] = 0


class PayrollUpdate(BaseModel):
    hourly_rate: Optional[Decimal] = None
    overtime_rate: Optional[Decimal] = None
    deductions: Optional[Decimal] = None
    status: Optional[str] = None


class PayrollResponse(BaseModel):
    id: str
    employee_id: str
    month: int
    year: int
    salary: Decimal
    hours_worked: Decimal
    hourly_rate: Decimal
    overtime_hours: Decimal
    overtime_rate: Decimal
    deductions: Decimal
    net_salary: Decimal
    status: str
    processed_at: datetime
    created_at: datetime

    class Config:
        from_attributes = True


class PayrollWithEmployee(PayrollResponse):
    employee_name: str
    employee_email: str
    employee_department: Optional[str]


class PayrollCalculationRequest(BaseModel):
    employee_id: str
    month: int
    year: int
    hourly_rate: Decimal
    overtime_rate: Optional[Decimal] = None
    deductions: Optional[Decimal] = 0


class PayrollCalculationResponse(BaseModel):
    employee_id: str
    employee_name: str
    month: int
    year: int
    hours_worked: Decimal
    overtime_hours: Decimal
    hourly_rate: Decimal
    overtime_rate: Decimal
    base_salary: Decimal
    overtime_pay: Decimal
    gross_salary: Decimal
    deductions: Decimal
    net_salary: Decimal
    attendance_days: int


class PayrollReportRequest(BaseModel):
    start_month: int
    start_year: int
    end_month: int
    end_year: int
    employee_id: Optional[str] = None
