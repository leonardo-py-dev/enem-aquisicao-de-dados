import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Ensure root directory resolution
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Pipeline Global Configuration Settings."""

    # Database
    DATABASE_URL: str = Field(
        default=f"sqlite:///{BASE_DIR / 'data' / 'enem_pipeline.db'}",
        description="Database connection string (SQLite local or PostgreSQL/Supabase)",
    )

    # Supabase optional settings
    SUPABASE_PROJECT_REF: str = Field(default="", description="Supabase Project Reference")
    SUPABASE_URL: str = Field(default="", description="Supabase API URL")
    SUPABASE_ANON_KEY: str = Field(default="", description="Supabase Anon/Public Key")

    # Storage paths
    RAW_DATA_DIR: Path = Field(default=BASE_DIR / "data" / "raw")
    PROCESSED_DATA_DIR: Path = Field(default=BASE_DIR / "data" / "processed")

    # Network / Scraping controls
    DEFAULT_HTTP_TIMEOUT: float = Field(default=30.0, description="HTTP timeout in seconds")
    DEFAULT_MAX_RETRIES: int = Field(default=3, description="Max retry attempts for collectors")
    USER_AGENT: str = Field(
        default="EnemPredictorDataPipeline/1.0 (+https://github.com/enem-predictor)",
        description="User-Agent header for HTTP requests",
    )

    # Logging
    LOG_LEVEL: str = Field(default="INFO")

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

# Ensure data directories exist
settings.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
