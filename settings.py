"""Application configuration module.

This module centralizes all configuration settings for the PocketSomm backend.
It reads environment variables (optionally from a `.env` file) and exposes
constants that other modules can import. This makes it easy to adjust
deployment parameters (data directory, environment, CORS origins) without
modifying source code.

Usage:
    from settings import OPENAI_API_KEY, DATA_DIR, ENV, CORS_ORIGINS
"""

import os
from dotenv import load_dotenv

# Load variables from a .env file if present (noop if missing)
load_dotenv()

# OpenAI API key (required)
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY must be set in the environment or .env file"
    )

# Base directory for persistent data (default: "data"). This directory will
# contain user profiles and the wines database. Changing this value lets you
# keep data in a different location for development, testing, or production.
DATA_DIR: str = os.getenv("POCKETSOMM_DATA_DIR", "data")

# Environment mode (default: "development"). Not currently used by the code
# but reserved for future conditional logic (e.g. feature toggles).
ENV: str = os.getenv("POCKETSOMM_ENV", "development")

# Allowed CORS origins. Comma-separated list of origins. A single "*" will
# allow all origins (useful for local development). Empty or missing value
# defaults to ["*"] for convenience.
cors_origins_env = os.getenv("POCKETSOMM_CORS_ORIGINS", "*")
CORS_ORIGINS: list[str] = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]
if not CORS_ORIGINS:
    CORS_ORIGINS = ["*"]
