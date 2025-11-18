"""
MyFreeScoreNow API Integration Routes
"""
from fastapi import APIRouter, Depends, HTTPException, status
from typing import Dict, Any
import httpx
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.schemas import (
    MFSN3BReportRequest, MFSNEnrollmentRequest, 
    MFSNSnapshotEnrollRequest, SuccessResponse, ErrorResponse
)
from api.routes.auth import get_current_user
from config.settings import settings

router = APIRouter()


class MFSNClient:
    """MyFreeScoreNow API Client"""
    
    def __init__(self):
        self.base_url = settings.MFSN_API_URL
        self.auth_token = None
    
    async def login(self) -> str:
        """Login to MFSN API and get auth token"""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/auth/login",
                    json={
                        "email": settings.MFSN_EMAIL,
                        "password": settings.MFSN_PASSWORD
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                if data.get("success"):
                    self.auth_token = data["data"]["token"]
                    return self.auth_token
                else:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="MFSN authentication failed"
                    )
            except httpx.HTTPError as e:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"MFSN API connection error: {str(e)}"
                )
    
    async def get_3b_report_json(self, username: str, password: str) -> Dict[str, Any]:
        """Get 3-bureau credit report in JSON format"""
        if not self.auth_token:
            await self.login()
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/auth/3B/report.json",
                    headers={"Authorization": f"Bearer {self.auth_token}"},
                    json={
                        "username": username,
                        "password": password
                    }
                )
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPError as e:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Failed to retrieve credit report: {str(e)}"
                )
    
    async def get_epic_pro_report_json(self, username: str, password: str) -> Dict[str, Any]:
        """Get Epic Pro credit report in JSON format"""
        if not self.auth_token:
            await self.login()
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/auth/v2/3B/epic/report.json",
                    headers={"Authorization": f"Bearer {self.auth_token}"},
                    json={
                        "username": username,
                        "password": password
                    }
                )
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPError as e:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Failed to retrieve Epic Pro report: {str(e)}"
                )
    
    async def enroll_user(self, enrollment_data: MFSNEnrollmentRequest) -> Dict[str, Any]:
        """Start enrollment process"""
        if not self.auth_token:
            await self.login()
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/auth/enroll/start",
                    headers={"Authorization": f"Bearer {self.auth_token}"},
                    json=enrollment_data.dict()
                )
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPError as e:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Enrollment failed: {str(e)}"
                )
    
    async def snapshot_enroll(self, snapshot_data: MFSNSnapshotEnrollRequest, 
                            report_type: str = "credit") -> Dict[str, Any]:
        """Snapshot enrollment (credit or funding)"""
        if not self.auth_token:
            await self.login()
        
        endpoint = f"/auth/snapshot/{report_type}/enroll"
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}{endpoint}",
                    headers={"Authorization": f"Bearer {self.auth_token}"},
                    json=snapshot_data.dict()
                )
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPError as e:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Snapshot enrollment failed: {str(e)}"
                )


# Initialize MFSN client
mfsn_client = MFSNClient()


@router.post("/3b-report", response_model=SuccessResponse)
async def get_3b_credit_report(
    request: MFSN3BReportRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Get 3-bureau credit report from MyFreeScoreNow
    
    - **username**: MFSN account email
    - **password**: MFSN account password
    
    Returns complete credit report with:
    - Credit scores from all 3 bureaus
    - Trade lines (credit accounts)
    - Inquiries
    - Public records
    - Personal information
    """
    try:
        report_data = await mfsn_client.get_3b_report_json(
            request.username,
            request.password
        )
        
        # TODO: Save report to database
        # TODO: Index in vector database for RAG
        # TODO: Queue for AI agent analysis
        
        return SuccessResponse(
            success=True,
            message="Credit report retrieved successfully",
            data=report_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


@router.post("/epic-report", response_model=SuccessResponse)
async def get_epic_pro_report(
    request: MFSN3BReportRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Get Epic Pro credit report from MyFreeScoreNow
    
    Enhanced version with additional features
    """
    try:
        report_data = await mfsn_client.get_epic_pro_report_json(
            request.username,
            request.password
        )
        
        return SuccessResponse(
            success=True,
            message="Epic Pro report retrieved successfully",
            data=report_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


@router.post("/enroll", response_model=SuccessResponse)
async def enroll_new_user(
    enrollment: MFSNEnrollmentRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Start MFSN enrollment process
    
    Initiates the full enrollment workflow including:
    - User registration
    - Identity verification questions
    - Credit card setup
    - Security questions
    """
    try:
        enrollment_result = await mfsn_client.enroll_user(enrollment)
        
        # TODO: Track affiliate conversion
        # TODO: Save enrollment data
        
        return SuccessResponse(
            success=True,
            message="Enrollment initiated successfully",
            data=enrollment_result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


@router.post("/snapshot/enroll/{report_type}", response_model=SuccessResponse)
async def snapshot_enrollment(
    report_type: str,
    snapshot: MFSNSnapshotEnrollRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Quick snapshot enrollment
    
    - **report_type**: "credit" or "funding"
    
    Faster enrollment process for instant credit checks
    """
    if report_type not in ["credit", "funding"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="report_type must be 'credit' or 'funding'"
        )
    
    try:
        snapshot_result = await mfsn_client.snapshot_enroll(snapshot, report_type)
        
        return SuccessResponse(
            success=True,
            message=f"Snapshot {report_type} enrollment successful",
            data=snapshot_result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )


@router.get("/test-connection", response_model=SuccessResponse)
async def test_mfsn_connection(current_user: dict = Depends(get_current_user)):
    """
    Test MFSN API connection and authentication
    """
    try:
        token = await mfsn_client.login()
        
        return SuccessResponse(
            success=True,
            message="MFSN API connection successful",
            data={"authenticated": True, "token_length": len(token)}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Connection test failed: {str(e)}"
        )
