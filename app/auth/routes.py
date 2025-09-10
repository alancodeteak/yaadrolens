from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.auth.schemas import UserCreate, UserLogin, UserResponse, Token, RefreshTokenRequest
from app.auth.service import AuthService
from app.core.security import verify_token
from app.core.dependencies import get_current_user

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user (admin/HR account)."""
    auth_service = AuthService(db)
    user = auth_service.create_user(user_data)
    return user


@router.post("/login", response_model=Token)
async def login_user(login_data: UserLogin, db: Session = Depends(get_db)):
    """Login user and return JWT tokens."""
    auth_service = AuthService(db)
    user = auth_service.authenticate_user(login_data)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    tokens = auth_service.create_tokens(user)
    return tokens


@router.post("/refresh", response_model=Token)
async def refresh_token(request: RefreshTokenRequest, db: Session = Depends(get_db)):
    """Refresh access token using refresh token."""
    try:
        # Verify refresh token
        payload = verify_token(request.refresh_token, "refresh")
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Get user
        auth_service = AuthService(db)
        user = auth_service.get_user_by_id(user_id)
        
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Create new tokens
        tokens = auth_service.create_tokens(user)
        return tokens
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user = Depends(get_current_user)):
    """Get current user information."""
    return current_user
