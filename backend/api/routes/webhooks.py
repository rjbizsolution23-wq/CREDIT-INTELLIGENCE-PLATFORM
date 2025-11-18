"""
Webhook Routes
Handle callbacks from MFSN, Stripe, and other services
"""
from fastapi import APIRouter, Request, HTTPException, status, Header
from typing import Optional
import hmac
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.schemas import SuccessResponse
from config.settings import settings

router = APIRouter()


def verify_stripe_signature(payload: bytes, signature: str) -> bool:
    """Verify Stripe webhook signature"""
    if not settings.STRIPE_WEBHOOK_SECRET:
        return True  # Skip verification in development
    
    try:
        expected_signature = hmac.new(
            settings.STRIPE_WEBHOOK_SECRET.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)
    except Exception:
        return False


@router.post("/stripe", response_model=SuccessResponse)
async def stripe_webhook(
    request: Request,
    stripe_signature: Optional[str] = Header(None, alias="Stripe-Signature")
):
    """
    Handle Stripe webhook events
    
    Events handled:
    - checkout.session.completed
    - customer.subscription.created
    - customer.subscription.updated
    - customer.subscription.deleted
    - invoice.payment_succeeded
    - invoice.payment_failed
    """
    payload = await request.body()
    
    # Verify signature in production
    if settings.ENVIRONMENT == "production":
        if not stripe_signature or not verify_stripe_signature(payload, stripe_signature):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid signature"
            )
    
    event = await request.json()
    event_type = event.get("type")
    
    if event_type == "checkout.session.completed":
        # Handle successful checkout
        session = event["data"]["object"]
        customer_email = session.get("customer_email")
        amount = session.get("amount_total", 0) / 100
        
        # TODO: Provision user access
        # TODO: Send welcome email
        # TODO: Track conversion
        
        print(f"Checkout completed: {customer_email} - ${amount}")
        
    elif event_type == "customer.subscription.created":
        # Handle new subscription
        subscription = event["data"]["object"]
        customer_id = subscription.get("customer")
        
        # TODO: Activate subscription features
        # TODO: Update user tier
        
        print(f"Subscription created: {customer_id}")
        
    elif event_type == "customer.subscription.updated":
        # Handle subscription update
        subscription = event["data"]["object"]
        customer_id = subscription.get("customer")
        status_sub = subscription.get("status")
        
        # TODO: Update subscription status
        
        print(f"Subscription updated: {customer_id} - {status_sub}")
        
    elif event_type == "customer.subscription.deleted":
        # Handle subscription cancellation
        subscription = event["data"]["object"]
        customer_id = subscription.get("customer")
        
        # TODO: Revoke access
        # TODO: Send cancellation email
        
        print(f"Subscription deleted: {customer_id}")
        
    elif event_type == "invoice.payment_succeeded":
        # Handle successful payment
        invoice = event["data"]["object"]
        customer_id = invoice.get("customer")
        amount = invoice.get("amount_paid", 0) / 100
        
        # TODO: Extend subscription period
        # TODO: Send receipt
        
        print(f"Payment succeeded: {customer_id} - ${amount}")
        
    elif event_type == "invoice.payment_failed":
        # Handle failed payment
        invoice = event["data"]["object"]
        customer_id = invoice.get("customer")
        
        # TODO: Send payment failure notification
        # TODO: Implement retry logic
        
        print(f"Payment failed: {customer_id}")
    
    return SuccessResponse(
        success=True,
        message=f"Webhook processed: {event_type}"
    )


@router.post("/mfsn", response_model=SuccessResponse)
async def mfsn_webhook(request: Request):
    """
    Handle MyFreeScoreNow webhook events
    
    Events:
    - enrollment_completed
    - report_ready
    - subscription_status_changed
    """
    event = await request.json()
    event_type = event.get("event_type")
    
    if event_type == "enrollment_completed":
        user_id = event.get("user_id")
        data = event.get("data", {})
        
        # TODO: Track affiliate conversion
        # TODO: Update user status
        # TODO: Send confirmation email
        
        print(f"MFSN enrollment completed: {user_id}")
        
    elif event_type == "report_ready":
        user_id = event.get("user_id")
        report_id = event.get("data", {}).get("report_id")
        
        # TODO: Queue AI analysis
        # TODO: Notify user
        
        print(f"MFSN report ready: {user_id} - {report_id}")
        
    elif event_type == "subscription_status_changed":
        user_id = event.get("user_id")
        new_status = event.get("data", {}).get("status")
        
        # TODO: Update subscription status
        
        print(f"MFSN subscription status changed: {user_id} - {new_status}")
    
    return SuccessResponse(
        success=True,
        message=f"MFSN webhook processed: {event_type}"
    )


@router.post("/test", response_model=SuccessResponse)
async def test_webhook(request: Request):
    """
    Test webhook endpoint for development
    """
    payload = await request.json()
    
    return SuccessResponse(
        success=True,
        message="Test webhook received",
        data=payload
    )
