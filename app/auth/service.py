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
from app.core.service_base import BaseService
from app.core.exceptions import (
    ResourceAlreadyExistsError,
    AuthenticationError,
    ResourceNotFoundError,
    ResourceInactiveError
)
from typing import Optional


class AuthService(BaseService):
    def __init__(self, db: Session):
        super().__init__(db)
    
    def create_user(self, user_data: UserCreate) -> User:
        """Create a new user with hashed password."""
        # Validate required fields
        self.validate_required_fields(
            {"email": user_data.email, "password": user_data.password},
            ["email", "password"]
        )
        
        # Check if user already exists
        self.check_unique_constraint(User, "email", user_data.email, "User")
        
        # Create new user
        hashed_password = get_password_hash(user_data.password)
        db_user = User(
            email=user_data.email,
            password_hash=hashed_password,
            role=user_data.role
        )
        
        self.db.add(db_user)
        self.safe_commit("Error creating user")
        self.db.refresh(db_user)
        
        self.log_service_action("create_user", "User", str(db_user.id))
        return db_user
    
    def authenticate_user(self, login_data: UserLogin) -> User:
        """Authenticate user with email and password."""
        # Validate required fields
        self.validate_required_fields(
            {"email": login_data.email, "password": login_data.password},
            ["email", "password"]
        )
        
        user = self.db.query(User).filter(User.email == login_data.email).first()
        
        if not user:
            self.log_service_action("failed_login_attempt", extra_data={"email": login_data.email, "reason": "user_not_found"})
            raise AuthenticationError("Invalid email or password")
        
        if not user.is_active:
            self.log_service_action("failed_login_attempt", extra_data={"email": login_data.email, "reason": "user_inactive"})
            raise ResourceInactiveError("User", str(user.id))
        
        if not verify_password(login_data.password, user.password_hash):
            self.log_service_action("failed_login_attempt", extra_data={"email": login_data.email, "reason": "invalid_password"})
            raise AuthenticationError("Invalid email or password")
        
        self.log_service_action("successful_login", "User", str(user.id))
        return user
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        try:
            return self.get_or_404(User, user_id, "User")
        except ResourceNotFoundError:
            return None  # For backward compatibility
    
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
