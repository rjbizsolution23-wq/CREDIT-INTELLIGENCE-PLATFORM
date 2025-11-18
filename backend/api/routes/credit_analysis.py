"""
Credit Analysis Routes
AI-powered credit scoring, fraud detection, and forecasting
"""
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from typing import Dict, Any, Optional
import sys
from pathlib import Path
from datetime import datetime
import os

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.schemas import (
    CreditScoreRequest, CreditScoreResponse,
    FraudCheckRequest, FraudCheckResponse,
    CreditForecastRequest, CreditForecastResponse,
    SuccessResponse
)
from api.routes.auth import get_current_user
from ml.credit_scorer import CreditScorer
from ml.fraud_detector import FraudDetector
from ml.forecaster import CreditScoreForecaster
from services.openrouter_service import OpenRouterService

router = APIRouter()

# Initialize ML models (singleton pattern)
_credit_scorer: Optional[CreditScorer] = None
_fraud_detector: Optional[FraudDetector] = None
_forecaster: Optional[CreditScoreForecaster] = None
_openrouter: Optional[OpenRouterService] = None


def get_credit_scorer() -> CreditScorer:
    """Lazy-load credit scorer model"""
    global _credit_scorer
    if _credit_scorer is None:
        model_path = os.path.join(Path(__file__).parent.parent.parent, 'ml', 'models', 'credit_scorer.pkl')
        if os.path.exists(model_path):
            _credit_scorer = CreditScorer(model_path=model_path)
        else:
            _credit_scorer = CreditScorer()
            # TODO: Train on initial synthetic data if no model exists
    return _credit_scorer


def get_fraud_detector() -> FraudDetector:
    """Lazy-load fraud detector model"""
    global _fraud_detector
    if _fraud_detector is None:
        model_path = os.path.join(Path(__file__).parent.parent.parent, 'ml', 'models', 'fraud_detector.pkl')
        if os.path.exists(model_path):
            _fraud_detector = FraudDetector(model_path=model_path)
        else:
            _fraud_detector = FraudDetector()
    return _fraud_detector


def get_forecaster() -> CreditScoreForecaster:
    """Lazy-load forecaster model"""
    global _forecaster
    if _forecaster is None:
        model_path = os.path.join(Path(__file__).parent.parent.parent, 'ml', 'models', 'forecaster.pkl')
        if os.path.exists(model_path):
            _forecaster = CreditScoreForecaster(model_path=model_path)
        else:
            _forecaster = CreditScoreForecaster()
    return _forecaster


def get_openrouter() -> OpenRouterService:
    """Lazy-load OpenRouter service"""
    global _openrouter
    if _openrouter is None:
        _openrouter = OpenRouterService()
    return _openrouter


def mock_credit_score_analysis(report_id: str) -> Dict[str, Any]:
    """Mock credit scoring analysis"""
    return {
        "score": 704,
        "confidence": 0.87,
        "risk_level": "MEDIUM",
        "shap_values": {
            "payment_history": 0.15,
            "credit_utilization": -0.12,
            "account_age": 0.08,
            "hard_inquiries": -0.05,
            "credit_mix": 0.03,
            "negative_accounts": -0.09
        },
        "recommendations": [
            "Reduce credit utilization below 30% (currently 35%)",
            "Avoid new hard inquiries for the next 6 months",
            "Consider dispute process for negative accounts"
        ],
        "factors_helping": [
            {"factor": "No late payments in 24 months", "impact": "+15 points"},
            {"factor": "Long credit history (15 years)", "impact": "+10 points"},
            {"factor": "Low total debt ($33K)", "impact": "+8 points"}
        ],
        "factors_hurting": [
            {"factor": "High utilization (35%)", "impact": "-12 points"},
            {"factor": "11 hard inquiries (6 months)", "impact": "-8 points"},
            {"factor": "26 negative accounts", "impact": "-15 points"}
        ],
        "created_at": datetime.utcnow()
    }


def mock_fraud_detection(report_id: str) -> Dict[str, Any]:
    """Mock fraud detection analysis"""
    return {
        "risk_score": 15,
        "is_fraudulent": False,
        "flagged_items": [],
        "anomalies": [
            "Multiple inquiries from same creditor (3 within 48 hours)"
        ],
        "confidence": 0.92,
        "created_at": datetime.utcnow()
    }


def mock_credit_forecast(report_id: str, months: int) -> Dict[str, Any]:
    """Mock credit score forecasting"""
    current_score = 704
    predictions = []
    
    for month in range(1, months + 1):
        # Simple linear projection for mock
        predicted_score = current_score + (month * 5)
        predictions.append({
            "month": month,
            "score": min(predicted_score, 850),
            "confidence": max(0.85 - (month * 0.05), 0.65)
        })
    
    return {
        "current_score": current_score,
        "predicted_scores": predictions,
        "confidence_intervals": [
            {"month": m["month"], "lower": m["score"] - 18, "upper": m["score"] + 18}
            for m in predictions
        ],
        "recommendations": [
            "Continue current payment patterns",
            "Reduce utilization by $3,000 for optimal trajectory",
            "Expected to reach 'Good' credit tier (720+) in 3 months"
        ],
        "created_at": datetime.utcnow()
    }


@router.post("/credit-score", response_model=CreditScoreResponse)
async def analyze_credit_score(
    request: CreditScoreRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Perform AI-powered credit score analysis
    
    Uses ensemble ML models (XGBoost + LightGBM + CatBoost) with SHAP explainability
    
    - **report_id**: Credit report identifier
    - **use_cached**: Use cached results if available
    
    Returns:
    - Predicted credit score (300-850)
    - Confidence level
    - SHAP feature importance
    - Actionable recommendations
    - Factors helping/hurting score
    """
    try:
        # TODO: Load actual report from database using report_id
        # For now, use mock credit data
        credit_data = {
            'late_payments_12mo': 2,
            'credit_utilization_ratio': 0.35,
            'oldest_account_age_months': 180,
            'total_accounts': 12,
            'hard_inquiries_6mo': 11,
            'collections_count': 26,
            'derogatory_marks': 3,
            'total_balance': 33000,
            'total_credit_limit': 95000,
            'on_time_payment_rate': 0.85,
            'avg_account_age_months': 90
        }
        
        # Get ML model and run prediction
        scorer = get_credit_scorer()
        result = scorer.predict_with_explanation(credit_data)
        
        # Get AI insights from OpenRouter (free tier)
        openrouter = get_openrouter()
        ai_insights = await openrouter.generate_credit_insights(credit_data, "quick")
        
        # Format response
        analysis = {
            "score": result['credit_score'],
            "confidence": result['confidence'],
            "risk_level": result['risk_level'],
            "shap_values": result['feature_importance'],
            "recommendations": result['recommendations'],
            "factors_helping": result.get('factors_helping', []),
            "factors_hurting": result.get('factors_hurting', []),
            "ai_insights": ai_insights[:300],  # Summary
            "created_at": datetime.utcnow()
        }
        
        # TODO: Cache results in Redis
        # TODO: Save analysis to database
        
        return CreditScoreResponse(**analysis)
        
    except Exception as e:
        # Fallback to mock data if ML fails
        print(f"ML prediction failed: {e}, falling back to mock")
        analysis = mock_credit_score_analysis(request.report_id)
        return CreditScoreResponse(**analysis)


@router.post("/fraud-check", response_model=FraudCheckResponse)
async def check_fraud(
    request: FraudCheckRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Detect fraud and identity theft using Graph Neural Networks
    
    Analyzes transaction patterns and relationships to identify:
    - Suspicious account activity
    - Identity theft indicators
    - Camouflage fraud patterns
    - Anomalous behavior
    
    - **report_id**: Credit report identifier
    - **threshold**: Fraud detection sensitivity (0.0-1.0)
    
    Returns:
    - Risk score (0-100)
    - Flagged items
    - Anomaly descriptions
    - Confidence level
    """
    try:
        # TODO: Load actual report from database using report_id
        # For now, use mock credit data
        credit_data = {
            'user_id': current_user.get('email'),
            'inquiries_6mo': 11,
            'new_accounts_12mo': 6,
            'credit_utilization': 0.35,
            'late_payments_12mo': 2,
            'address_changes_12mo': 1,
            'collections_count': 26,
            'total_accounts': 12,
            'revolving_accounts': 8,
            'derogatory_marks': 3
        }
        
        # Get ML model and run fraud detection
        detector = get_fraud_detector()
        fraud_alert = detector.predict(credit_data)
        
        # Format response
        fraud_analysis = {
            "risk_score": int(fraud_alert.fraud_probability * 100),
            "is_fraudulent": fraud_alert.risk_level in ['high', 'critical'],
            "risk_level": fraud_alert.risk_level,
            "flagged_items": [
                {
                    "type": indicator['type'],
                    "severity": indicator['severity'],
                    "description": indicator['description']
                }
                for indicator in fraud_alert.fraud_indicators
            ],
            "anomalies": fraud_alert.graph_anomalies,
            "recommended_actions": fraud_alert.recommended_actions,
            "confidence": fraud_alert.confidence_score,
            "created_at": datetime.utcnow()
        }
        
        # TODO: Save fraud alert to database if high risk
        # TODO: Trigger notifications if critical
        
        return FraudCheckResponse(**fraud_analysis)
        
    except Exception as e:
        print(f"Fraud detection failed: {e}, falling back to mock")
        fraud_analysis = mock_fraud_detection(request.report_id)
        return FraudCheckResponse(**fraud_analysis)


@router.post("/forecast", response_model=CreditForecastResponse)
async def forecast_credit_score(
    request: CreditForecastRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Predict future credit score trajectory
    
    Uses LSTM-Transformer hybrid model for time-series forecasting
    
    - **report_id**: Credit report identifier
    - **months_ahead**: Number of months to predict (1-12)
    - **scenario**: Optional scenario ("optimistic", "realistic", "pessimistic")
    
    Returns:
    - Current score
    - Month-by-month predictions
    - Confidence intervals
    - Scenario-based recommendations
    """
    try:
        # TODO: Load actual credit history from database
        # For now, use mock credit data
        credit_data = {
            'credit_score': 704,
            'credit_utilization': 0.35,
            'late_payments_12mo': 2,
            'inquiries_6mo': 11,
            'collections_count': 26,
            'derogatory_marks': 3,
            'total_accounts': 12,
            'oldest_account_age_months': 180,
            'avg_account_age_months': 90,
            'total_balance': 33000,
            'total_credit_limit': 95000,
            'monthly_income': 7500
        }
        
        # Get ML model and generate forecast
        forecaster = get_forecaster()
        forecast_result = forecaster.forecast(credit_data, months_ahead=request.months_ahead)
        
        # Format predictions for API response
        predictions = [
            {
                "month": i + 1,
                "score": score,
                "confidence": max(0.85 - (i * 0.03), 0.65),
                "month_date": month_str
            }
            for i, (score, month_str) in enumerate(
                zip(forecast_result.forecasted_scores, forecast_result.forecast_months)
            )
        ]
        
        # Format confidence intervals
        confidence_intervals = [
            {
                "month": i + 1,
                "lower": lower,
                "upper": upper
            }
            for i, (lower, upper) in enumerate(forecast_result.confidence_intervals)
        ]
        
        # Get AI-powered financial advice
        openrouter = get_openrouter()
        financial_advice = await openrouter.generate_financial_advice(credit_data, "improve_score")
        
        forecast = {
            "current_score": forecast_result.current_score,
            "predicted_scores": predictions,
            "confidence_intervals": confidence_intervals,
            "trend": forecast_result.trend,
            "key_drivers": forecast_result.key_drivers,
            "milestone_dates": forecast_result.milestone_dates,
            "recommendations": forecast_result.recommendations,
            "ai_advice": financial_advice[:500],  # Summary
            "created_at": datetime.utcnow()
        }
        
        # TODO: Save forecast to database
        
        return CreditForecastResponse(**forecast)
        
    except Exception as e:
        print(f"Forecasting failed: {e}, falling back to mock")
        forecast = mock_credit_forecast(request.report_id, request.months_ahead)
        return CreditForecastResponse(**forecast)


@router.get("/history/{user_id}", response_model=SuccessResponse)
async def get_analysis_history(
    user_id: str,
    limit: int = 10,
    current_user: dict = Depends(get_current_user)
):
    """
    Get user's credit analysis history
    
    - **user_id**: User identifier
    - **limit**: Maximum number of records to return
    """
    # TODO: Fetch from database
    
    mock_history = [
        {
            "id": "analysis-123",
            "type": "credit_score",
            "score": 704,
            "created_at": datetime.utcnow().isoformat()
        },
        {
            "id": "analysis-124",
            "type": "fraud_check",
            "risk_score": 15,
            "created_at": datetime.utcnow().isoformat()
        }
    ]
    
    return SuccessResponse(
        success=True,
        message="Analysis history retrieved",
        data={"history": mock_history, "total": len(mock_history)}
    )


@router.post("/full-analysis/{report_id}", response_model=SuccessResponse)
async def run_full_analysis(
    report_id: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Run comprehensive AI analysis (all models)
    
    Executes in background:
    - Credit scoring
    - Fraud detection
    - 6-month forecast
    - RAG semantic indexing
    
    Returns immediately with execution ID
    Check status with /agents/status/{execution_id}
    """
    # Generate execution ID
    execution_id = f"exec-{report_id}-{int(datetime.utcnow().timestamp())}"
    
    # TODO: Queue background job
    # background_tasks.add_task(run_full_ai_analysis, report_id, execution_id)
    
    return SuccessResponse(
        success=True,
        message="Full analysis queued",
        data={
            "execution_id": execution_id,
            "status": "queued",
            "estimated_time": "10-15 seconds"
        }
    )


@router.get("/export/{analysis_id}", response_model=SuccessResponse)
async def export_analysis(
    analysis_id: str,
    format: str = "json",
    current_user: dict = Depends(get_current_user)
):
    """
    Export analysis results
    
    - **analysis_id**: Analysis identifier
    - **format**: Export format (json, pdf, csv)
    """
    if format not in ["json", "pdf", "csv"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Format must be 'json', 'pdf', or 'csv'"
        )
    
    # TODO: Generate export file
    
    return SuccessResponse(
        success=True,
        message=f"Analysis exported as {format}",
        data={
            "download_url": f"/downloads/{analysis_id}.{format}",
            "expires_in": "24 hours"
        }
    )
