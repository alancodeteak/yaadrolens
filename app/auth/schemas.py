from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime
from uuid import UUID


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    role: str = "admin"


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: UUID
    email: str
    role: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: Optional[str] = None


class RefreshTokenRequest(BaseModel):
    refresh_token: str
