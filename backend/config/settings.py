"""
Application configuration settings
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # App Info
    APP_NAME: str = "Credit Intelligence Platform"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # API
    API_PREFIX: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/credit_intel"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_CACHE_TTL: int = 3600  # 1 hour
    
    # MyFreeScoreNow API
    MFSN_API_URL: str = "https://api.myfreescorenow.com/api"
    MFSN_EMAIL: str = "rickjefferson@rickjeffersonsolutions.com"
    MFSN_PASSWORD: str = "Nadia112318$"
    MFSN_AFFILIATE_ID: str = "RickJeffersonSolutions"
    MFSN_DEFAULT_PID: str = "49914"
    
    # OpenAI
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-large"
    
    # Pinecone
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: str = "us-west1-gcp"
    PINECONE_INDEX_NAME: str = "credit-intelligence"
    
    # Anthropic (for AutoGen)
    ANTHROPIC_API_KEY: Optional[str] = None
    
    # Model Paths
    MODEL_PATH: str = "./models"
    CREDIT_SCORING_MODEL: str = "credit_scorer_ensemble.pkl"
    FRAUD_DETECTION_MODEL: str = "fraud_detector_gnn.pt"
    FORECASTING_MODEL: str = "credit_forecast_transformer.pt"
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 100
    RATE_LIMIT_PER_HOUR: int = 1000
    
    # CORS
    ALLOWED_ORIGINS: list = [
        "http://localhost:8501",  # Streamlit default
        "http://localhost:3000",
        "http://localhost:8000",
    ]
    
    # Stripe
    STRIPE_API_KEY: Optional[str] = None
    STRIPE_WEBHOOK_SECRET: Optional[str] = None
    STRIPE_PRICE_STARTER: str = "price_starter_monthly"
    STRIPE_PRICE_PRO: str = "price_pro_monthly"
    STRIPE_PRICE_ENTERPRISE: str = "price_enterprise_monthly"
    
    # Monitoring
    SENTRY_DSN: Optional[str] = None
    PROMETHEUS_ENABLED: bool = True
    
    # Encryption (for storing MFSN credentials)
    ENCRYPTION_KEY: str = "change-this-to-32-byte-key-in-production"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields from .env


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


settings = get_settings()
