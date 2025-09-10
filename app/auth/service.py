from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status
from app.auth.models import User
from app.auth.schemas import UserCreate, UserLogin
from app.core.security import (
    get_password_hash, 
    verify_password, 
    create_access_token, 
    create_refresh_token
)
from typing import Optional


class AuthService:
    def __init__(self, db: Session):
        self.db = db
    
    def create_user(self, user_data: UserCreate) -> User:
        """Create a new user with hashed password."""
        try:
            # Check if user already exists
            existing_user = self.db.query(User).filter(User.email == user_data.email).first()
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
            
            # Create new user
            hashed_password = get_password_hash(user_data.password)
            db_user = User(
                email=user_data.email,
                password_hash=hashed_password,
                role=user_data.role
            )
            
            self.db.add(db_user)
            self.db.commit()
            self.db.refresh(db_user)
            
            return db_user
            
        except IntegrityError:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        except Exception as e:
            self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error creating user: {str(e)}"
            )
    
    def authenticate_user(self, login_data: UserLogin) -> Optional[User]:
        """Authenticate user with email and password."""
        user = self.db.query(User).filter(User.email == login_data.email).first()
        
        if not user:
            return None
        
        if not verify_password(login_data.password, user.password_hash):
            return None
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Inactive user"
            )
        
        return user
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.db.query(User).filter(User.id == user_id).first()
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        return self.db.query(User).filter(User.email == email).first()
    
    def create_tokens(self, user: User) -> dict:
        """Create access and refresh tokens for user."""
        access_token = create_access_token(data={"sub": str(user.id)})
        refresh_token = create_refresh_token(data={"sub": str(user.id)})
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }
