"""
FLIT ML API Main Application

FastAPI application providing ML inference endpoints for the FLIT ecosystem.
Supports BNPL risk assessment with production-ready performance and monitoring.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
from typing import Dict
import uvicorn

from flit_ml.api.bnpl_endpoints import router as bnpl_router


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Initialize FastAPI application
app = FastAPI(
    title="FLIT ML API",
    description="Machine Learning inference endpoints for financial risk assessment",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add request processing time header for monitoring."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))
    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled error in {request.method} {request.url.path}: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again.",
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )


# Include routers
app.include_router(bnpl_router)


# Root endpoint
@app.get("/")
async def root() -> Dict:
    """Root endpoint with API information."""
    return {
        "service": "FLIT ML API",
        "version": "0.1.0",
        "status": "operational",
        "endpoints": {
            "bnpl_risk_assessment": "/v1/bnpl/risk-assessment",
            "health_check": "/v1/bnpl/health",
            "model_info": "/v1/bnpl/models/info",
            "documentation": "/docs"
        }
    }


# Health check endpoint
@app.get("/health")
async def health() -> Dict[str, str]:
    """Global health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "service": "flit-ml-api"
    }


# Main entry point
def main():
    """Main entry point for the API server."""
    logger.info("Starting FLIT ML API server...")

    uvicorn.run(
        "flit_ml.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()