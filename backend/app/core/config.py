"""
Configuration management for BSPAI Backend.

Uses Pydantic Settings for environment variable management.
All configuration values should come from environment variables.
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    APP_NAME: str = "BSPAI Backend"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    ENVIRONMENT: str = Field(default="development", description="Environment name")

    # Server
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")

    # Database
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/bspai",
        description="PostgreSQL async connection URL",
    )
    DATABASE_POOL_SIZE: int = Field(default=10, description="Connection pool size")
    DATABASE_MAX_OVERFLOW: int = Field(default=20, description="Max overflow connections")

    # ML Service
    ML_SERVICE_URL: str = Field(
        default="http://ml:8001",
        description="ML service base URL (use service name in Docker)",
    )
    ML_SERVICE_TIMEOUT: float = Field(default=30.0, description="ML service request timeout (seconds)")

    # RAG Service
    RAG_SERVICE_URL: str = Field(
        default="http://rag:8002",
        description="RAG service base URL (use service name in Docker)",
    )
    RAG_SERVICE_TIMEOUT: float = Field(default=60.0, description="RAG service request timeout (seconds)")

    # Security
    SECRET_KEY: str = Field(
        default="change-me-in-production",
        description="Secret key for JWT/signing (must be changed in production)",
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, description="Access token expiry")
    BSPAI_API_KEYS: str = Field(
        default="",
        description="Comma-separated API keys for X-API-Key auth. Empty = open "
                    "development mode (all /api except /api/health still rate-limited; "
                    "do NOT use empty in production).",
    )

    # CORS
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://frontend:3000"],
        description="Allowed CORS origins",
    )
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True)
    CORS_ALLOW_METHODS: List[str] = Field(default=["*"])
    CORS_ALLOW_HEADERS: List[str] = Field(default=["*"])

    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(default="json", description="Log format: json or text")

    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = Field(default=True, description="Enable rate limiting")
    RATE_LIMIT_REQUESTS: int = Field(default=100, description="Requests per window")
    RATE_LIMIT_WINDOW_SECONDS: int = Field(default=60, description="Rate limit window")

    # Feature flags
    ENABLE_MOCK_ML: bool = Field(default=False, description="Use mock ML service for testing")
    ENABLE_MOCK_RAG: bool = Field(default=False, description="Use mock RAG service for testing")
    ENABLE_MOCK_LLM: bool = Field(default=False, description="Use mock LLM for testing")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Export for convenience
settings = get_settings()