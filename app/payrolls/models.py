from sqlalchemy import Column, String, DateTime, ForeignKey, Integer, Numeric
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base, generate_uuid


class Payroll(Base):
    __tablename__ = "payrolls"

    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    employee_id = Column(UUID(as_uuid=True), ForeignKey("employees.id"), nullable=False)
    month = Column(Integer, nullable=False)  # 1-12
    year = Column(Integer, nullable=False)
    salary = Column(Numeric(10, 2), nullable=False)
    hours_worked = Column(Numeric(8, 2), default=0)
    hourly_rate = Column(Numeric(8, 2), default=0)
    overtime_hours = Column(Numeric(8, 2), default=0)
    overtime_rate = Column(Numeric(8, 2), default=0)
    deductions = Column(Numeric(10, 2), default=0)
    net_salary = Column(Numeric(10, 2), nullable=False)
    status = Column(String(20), default="pending")  # pending, approved, paid
    processed_at = Column(DateTime(timezone=True), server_default=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    employee = relationship("Employee", back_populates="payrolls")
