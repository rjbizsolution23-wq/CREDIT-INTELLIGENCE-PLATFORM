"""
FastAPI Main Application
Supreme Credit Intelligence Platform
"""
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from api.routes import auth, mfsn, credit_analysis, agents, webhooks


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Elite AI-powered credit intelligence system with multi-agent orchestration",
    docs_url=f"{settings.API_PREFIX}/docs",
    redoc_url=f"{settings.API_PREFIX}/redoc",
    openapi_url=f"{settings.API_PREFIX}/openapi.json"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compression Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Trusted Host Middleware (production)
if settings.ENVIRONMENT == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*.example.com", "example.com"]
    )

# Prometheus Metrics
if settings.PROMETHEUS_ENABLED:
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# Error Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Validation Error",
            "details": exc.errors()
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal Server Error",
            "message": str(exc) if settings.DEBUG else "An error occurred"
        }
    )

# Startup Event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    
    # TODO: Initialize database connection pool
    # TODO: Initialize Redis connection
    # TODO: Load ML models
    # TODO: Initialize vector database
    
    logger.info("Application started successfully")

# Shutdown Event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down application")
    
    # TODO: Close database connections
    # TODO: Close Redis connections
    # TODO: Cleanup resources

# Health Check
@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT
    }

# Serve static files
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Root Endpoint - Serve chat interface
@app.get("/", tags=["System"])
async def root():
    """Root endpoint - Chat interface"""
    chat_file = Path(__file__).parent.parent / "static" / "chat.html"
    if chat_file.exists():
        return FileResponse(str(chat_file))
    return {
        "message": "Credit Intelligence Platform API",
        "version": settings.APP_VERSION,
        "docs": f"{settings.API_PREFIX}/docs"
    }

# Preview page endpoint
@app.get("/preview", tags=["System"])
async def preview():
    """Preview page with stats"""
    preview_file = Path(__file__).parent.parent / "static" / "index.html"
    if preview_file.exists():
        return FileResponse(str(preview_file))
    return {"message": "Preview page not found"}

# Include Routers
app.include_router(auth.router, prefix=f"{settings.API_PREFIX}/auth", tags=["Authentication"])
app.include_router(mfsn.router, prefix=f"{settings.API_PREFIX}/mfsn", tags=["MyFreeScoreNow"])
app.include_router(credit_analysis.router, prefix=f"{settings.API_PREFIX}/analysis", tags=["Credit Analysis"])
app.include_router(agents.router, prefix=f"{settings.API_PREFIX}/agents", tags=["AI Agents"])
app.include_router(webhooks.router, prefix=f"{settings.API_PREFIX}/webhooks", tags=["Webhooks"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
