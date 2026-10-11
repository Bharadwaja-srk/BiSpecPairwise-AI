"""
Security utilities for BSPAI Backend.

Provides CORS configuration, rate limiting, and basic security middleware.
"""

import logging

from typing import List

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.core.config import get_settings

logger = logging.getLogger(__name__)

_OPEN_MODE_WARNED = False


def _configured_keys() -> list[str]:
    raw = get_settings().BSPAI_API_KEYS or ""
    return [k.strip() for k in raw.split(",") if k.strip()]


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Shared-secret API-key boundary (X-API-Key header).

    Real enforcement, no fake identity: when BSPAI_API_KEYS is configured,
    every /api request except /api/health (plus docs/openapi) needs a
    matching key (constant-time compare), else 401. When empty, the service
    runs in explicitly logged open development mode. Never hardcoded keys.
    """

    EXEMPT_PREFIXES = ("/api/health", "/docs", "/redoc", "/openapi.json")

    async def dispatch(self, request: Request, call_next):
        global _OPEN_MODE_WARNED
        if not request.url.path.startswith("/api") or any(
            request.url.path.startswith(p) for p in self.EXEMPT_PREFIXES
        ):
            return await call_next(request)
        keys = _configured_keys()
        if not keys:
            if not _OPEN_MODE_WARNED:
                logger.warning("BSPAI_API_KEYS empty: API running in OPEN development mode")
                _OPEN_MODE_WARNED = True
            return await call_next(request)
        import hashlib
        import secrets

        provided = request.headers.get("x-api-key", "")
        ok = any(secrets.compare_digest(provided, k) for k in keys)
        if not ok:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid or missing API key (X-API-Key header)",
                         "status_code": 401},
            )
        request.state.api_key_fingerprint = hashlib.sha256(provided.encode()).hexdigest()[:12]
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware.

    Note: For production, consider using Redis-based rate limiting.
    """

    def __init__(self, app: FastAPI, requests_per_window: int, window_seconds: int):
        super().__init__(app)
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}

    async def dispatch(self, request: Request, call_next):
        if not get_settings().RATE_LIMIT_ENABLED:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        import time
        now = time.time()

        # Clean old entries
        if client_ip in self._requests:
            self._requests[client_ip] = [
                t for t in self._requests[client_ip] if now - t < self.window_seconds
            ]
        else:
            self._requests[client_ip] = []

        # Check rate limit
        if len(self._requests[client_ip]) >= self.requests_per_window:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.requests_per_window} requests per {self.window_seconds} seconds",
                },
                headers={"Retry-After": str(self.window_seconds)},
            )

        # Record request
        self._requests[client_ip].append(now)

        return await call_next(request)


def setup_cors(app: FastAPI) -> None:
    """Configure CORS middleware."""
    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
    )


def setup_security_middleware(app: FastAPI) -> None:
    """Set up all security-related middleware."""
    setup_cors(app)

    settings = get_settings()
    if settings.RATE_LIMIT_ENABLED:
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_window=settings.RATE_LIMIT_REQUESTS,
            window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
        )
    app.add_middleware(ApiKeyMiddleware)


# Security headers middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # Don't add HSTS in development
        if get_settings().ENVIRONMENT == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response