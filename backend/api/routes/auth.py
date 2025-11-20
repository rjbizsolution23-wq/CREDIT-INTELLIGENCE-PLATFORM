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

# Password hashing - Using sha256 to avoid bcrypt 72-byte limit
import hashlib

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_PREFIX}/auth/login")


def hash_password(password: str) -> str:
    """Hash a password using SHA-256 (no length limit)"""
    # For development, use SHA-256 hash to avoid bcrypt 72-byte limit
    # In production, consider using argon2 which has no such limit
    salt = "rj_business_solutions_2025"  # In production, use random salt per user
    salted = f"{salt}{password}".encode('utf-8')
    return hashlib.sha256(salted).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against hash"""
    # Use same SHA-256 method as hashing
    return hash_password(plain_password) == hashed_password


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


@router.post("/register")
async def register(user: UserRegister):
    """
    Register a new user
    
    - **email**: Valid email address
    - **password**: Minimum 8 characters
    - **full_name**: User's full name
    - **phone**: Phone number
    - **date_of_birth**: Date of birth
    - **address**: Full address object
    - **ssn_last4**: Last 4 digits of SSN
    """
    # TODO: Check if user already exists in database
    # TODO: Save user to database
    
    # For now, return mock response with tokens
    hashed_pwd = hash_password(user.password)
    
    # Generate unique user ID
    import uuid
    user_id = str(uuid.uuid4())
    
    # Create tokens
    access_token = create_access_token(
        data={"sub": user.email, "user_id": user_id}
    )
    refresh_token = create_refresh_token(
        data={"sub": user.email, "user_id": user_id}
    )
    
    # Return response matching frontend expectations
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": {
            "id": user_id,
            "email": user.email,
            "name": user.full_name,
            "phone": user.phone if hasattr(user, 'phone') else None,
            "created_at": datetime.utcnow().isoformat()
        }
    }


@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login with email and password
    
    Returns access token, refresh token, and user data
    """
    # TODO: Verify user credentials from database
    # Mock authentication for now
    
    email = form_data.username
    password = form_data.password
    
    # Mock user data
    import uuid
    user_id = str(uuid.uuid4())
    
    # Create tokens
    access_token = create_access_token(
        data={"sub": email, "user_id": user_id}
    )
    refresh_token = create_refresh_token(
        data={"sub": email, "user_id": user_id}
    )
    
    # Return response matching frontend expectations
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": {
            "id": user_id,
            "email": email,
            "name": "Rick Jefferson",
            "created_at": datetime.utcnow().isoformat()
        }
    }


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
