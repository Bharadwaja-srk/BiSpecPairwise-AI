"""
BSPAI Backend - FastAPI Application.

Main entry point for the BiSpec Pairwise AI backend service.
Provides REST API for bispecific antibody target combination analysis.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import analysis, features, targets, ranking, reports
from app.core.config import get_settings
from app.core.security import setup_security_middleware, SecurityHeadersMiddleware
from app.database.connection import init_db, close_db, db_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup and shutdown."""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")

    # Initialize database
    await init_db()
    logger.info("Database initialized")

    # Warm up downstream service clients (non-fatal: services may start later)
    try:
        from app.services.ml_service import get_ml_service
        from app.services.rag_service import get_rag_service

        ml, rag = get_ml_service(), get_rag_service()
        logger.info(f"ML client ready (mock={type(ml).__name__}), "
                    f"RAG client ready (mock={type(rag).__name__})")
    except Exception as e:
        logger.warning(f"Downstream client warmup deferred: {e}")

    logger.info("Application startup complete")
    yield

    # Shutdown
    logger.info("Shutting down application...")
    await close_db()
    logger.info("Database connections closed")
    logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="BiSpec Pairwise AI (BSPAI) - Bispecific Antibody Target Combination Analysis",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        openapi_url="/openapi.json" if settings.DEBUG else None,
        lifespan=lifespan,
    )

    # Setup security middleware (CORS, rate limiting, security headers)
    setup_security_middleware(app)
    app.add_middleware(SecurityHeadersMiddleware)

    # Exception handlers
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail, "status_code": exc.status_code},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation error",
                "details": exc.errors(),
                "status_code": 422,
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "status_code": 500},
        )

    # Health check endpoint
    @app.get("/api/health", tags=["Health"])
    async def health_check() -> dict:
        """Health check endpoint for monitoring and load balancers."""
        # Check database connectivity
        db_healthy = False
        try:
            from sqlalchemy import text

            async with db_manager.session() as session:
                await session.execute(text("SELECT 1"))
            db_healthy = True
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")

        return {
            "status": "healthy" if db_healthy else "degraded",
            "service": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "database": "connected" if db_healthy else "disconnected",
        }

    # Include API routes
    app.include_router(analysis.router, prefix="/api", tags=["Analysis"])
    app.include_router(targets.router, prefix="/api", tags=["Targets"])
    app.include_router(ranking.router, prefix="/api", tags=["Ranking"])
    app.include_router(reports.router, prefix="/api", tags=["Reports"])
    app.include_router(features.router, prefix="/api", tags=["Features"])

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )