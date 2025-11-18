"""
Authentication Routes
JWT-based authentication with refresh tokens
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.schemas import (
    UserRegister, UserLogin, Token, UserResponse, 
    SuccessResponse, ErrorResponse
)
from config.settings import settings

router = APIRouter()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_PREFIX}/auth/login")


def hash_password(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against hash"""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """Get current user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        user_id: str = payload.get("user_id")
        token_type: str = payload.get("type")
        
        if email is None or user_id is None or token_type != "access":
            raise credentials_exception
            
        return {"email": email, "user_id": user_id}
        
    except JWTError:
        raise credentials_exception


@router.post("/register", response_model=SuccessResponse, status_code=status.HTTP_201_CREATED)
async def register(user: UserRegister):
    """
    Register a new user
    
    - **email**: Valid email address
    - **password**: Minimum 8 characters
    - **full_name**: User's full name
    - **affiliate_id**: Optional affiliate tracking ID
    """
    # TODO: Check if user already exists in database
    # TODO: Save user to database
    
    # For now, return mock response
    hashed_pwd = hash_password(user.password)
    
    return SuccessResponse(
        success=True,
        message="User registered successfully",
        data={
            "user_id": "mock-user-id-123",
            "email": user.email,
            "full_name": user.full_name
        }
    )


@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login with email and password
    
    Returns access token and refresh token
    """
    # TODO: Verify user credentials from database
    # Mock authentication for now
    
    email = form_data.username
    password = form_data.password
    
    # Mock user data
    user_id = "mock-user-id-123"
    
    # Create tokens
    access_token = create_access_token(
        data={"sub": email, "user_id": user_id}
    )
    refresh_token = create_refresh_token(
        data={"sub": email, "user_id": user_id}
    )
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer"
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(refresh_token: str):
    """
    Refresh access token using refresh token
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid refresh token"
    )
    
    try:
        payload = jwt.decode(refresh_token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        user_id: str = payload.get("user_id")
        token_type: str = payload.get("type")
        
        if email is None or user_id is None or token_type != "refresh":
            raise credentials_exception
        
        # Create new tokens
        new_access_token = create_access_token(
            data={"sub": email, "user_id": user_id}
        )
        new_refresh_token = create_refresh_token(
            data={"sub": email, "user_id": user_id}
        )
        
        return Token(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            token_type="bearer"
        )
        
    except JWTError:
        raise credentials_exception


@router.post("/logout", response_model=SuccessResponse)
async def logout(current_user: dict = Depends(get_current_user)):
    """
    Logout current user
    
    In a production system, you would:
    - Blacklist the token in Redis
    - Clear user session
    """
    return SuccessResponse(
        success=True,
        message="Successfully logged out"
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """
    Get current user information
    """
    # TODO: Fetch full user data from database
    
    # Mock response
    return UserResponse(
        id=current_user["user_id"],
        email=current_user["email"],
        full_name="Rick Jefferson",
        role="user",
        created_at=datetime.utcnow()
    )
