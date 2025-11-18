"""
Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ============= Authentication Schemas =============
class UserRole(str, Enum):
    USER = "user"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=2)
    affiliate_id: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    email: Optional[str] = None
    user_id: Optional[str] = None


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    role: UserRole
    created_at: datetime
    
    class Config:
        from_attributes = True


# ============= MFSN API Schemas =============
class MFSNLoginRequest(BaseModel):
    username: str
    password: str


class MFSN3BReportRequest(BaseModel):
    username: str
    password: str


class MFSNEnrollmentRequest(BaseModel):
    firstName: str
    lastName: str
    email: EmailStr
    password: str
    mobile: str
    streetAddress: str
    zip: str
    city: str
    state: str
    ssn: str
    dob: str  # Format: MM/DD/YYYY
    pid: str = "49914"
    aid: str = "RickJeffersonSolutions"
    sponsorCode: str = ""
    blackboxCode: Optional[str] = None


class MFSNSnapshotEnrollRequest(BaseModel):
    firstName: str
    lastName: str
    email: EmailStr
    password: str
    mobile: str
    streetAddress1: str
    city: str
    state: str
    zip: str
    dob: str  # Format: MM/DD/YYYY
    ssn: str
    aid: str = "RickJeffersonSolutions"
    type: str = "email"


# ============= Credit Analysis Schemas =============
class CreditScoreRequest(BaseModel):
    report_id: str
    use_cached: bool = True


class CreditScoreResponse(BaseModel):
    score: int = Field(..., ge=300, le=850)
    confidence: float = Field(..., ge=0, le=1)
    risk_level: str  # LOW, MEDIUM, HIGH
    shap_values: Dict[str, float]
    recommendations: List[str]
    factors_helping: List[Dict[str, Any]]
    factors_hurting: List[Dict[str, Any]]
    created_at: datetime


class FraudCheckRequest(BaseModel):
    report_id: str
    threshold: float = 0.5


class FraudCheckResponse(BaseModel):
    risk_score: int = Field(..., ge=0, le=100)
    is_fraudulent: bool
    flagged_items: List[Dict[str, Any]]
    anomalies: List[str]
    confidence: float
    created_at: datetime


class CreditForecastRequest(BaseModel):
    report_id: str
    months_ahead: int = Field(6, ge=1, le=12)
    scenario: Optional[str] = None  # "optimistic", "realistic", "pessimistic"


class CreditForecastResponse(BaseModel):
    current_score: int
    predicted_scores: List[Dict[str, Any]]  # [{month: 1, score: 720, confidence: 0.85}]
    confidence_intervals: List[Dict[str, Any]]
    recommendations: List[str]
    created_at: datetime


# ============= AI Agent Schemas =============
class AgentTask(str, Enum):
    CREDIT_SCORING = "credit_scoring"
    FRAUD_DETECTION = "fraud_detection"
    DISPUTE_GENERATION = "dispute_generation"
    FORECASTING = "forecasting"
    FULL_ANALYSIS = "full_analysis"


class AgentOrchestrationRequest(BaseModel):
    report_id: str
    tasks: List[AgentTask]
    priority: str = "normal"  # "low", "normal", "high"


class AgentOrchestrationResponse(BaseModel):
    execution_id: str
    status: str  # "queued", "running", "completed", "failed"
    results: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    created_at: datetime


class DisputeLetterRequest(BaseModel):
    report_id: str
    bureau: str  # "TRANSUNION", "EQUIFAX", "EXPERIAN"
    dispute_items: List[Dict[str, Any]]
    reason: str


class DisputeLetterResponse(BaseModel):
    letter_id: str
    bureau: str
    letter_content: str
    status: str
    created_at: datetime


# ============= RAG Search Schemas =============
class SemanticSearchRequest(BaseModel):
    query: str
    user_id: str
    top_k: int = Field(5, ge=1, le=20)
    filter_bureau: Optional[str] = None


class SemanticSearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_results: int
    query_time: float


# ============= Generic Response Schemas =============
class SuccessResponse(BaseModel):
    success: bool = True
    message: str
    data: Optional[Any] = None


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    details: Optional[Any] = None


# ============= Webhook Schemas =============
class StripeWebhookEvent(BaseModel):
    type: str
    data: Dict[str, Any]


class MFSNWebhookEvent(BaseModel):
    event_type: str
    user_id: str
    data: Dict[str, Any]
