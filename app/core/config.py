"""Application configuration.

Centralizes settings so nothing downstream reads os.environ directly.
Values come from environment variables or a .env file (gitignored).
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- App identity ---
    app_name: str = "MedVision AI"
    app_version: str = "0.1.0"
    # Non-diagnostic positioning is a project-wide invariant (DECISIONS.md D-04, D-05).
    disclaimer: str = (
        "MedVision AI is a research/educational prototype and is NOT a "
        "diagnostic tool. Outputs are not a substitute for evaluation by a "
        "qualified clinician."
    )
    environment: str = "development"

    # Public demo: when True, POST /auth/demo-login issues sessions for the
    # is_demo accounts (role-only, no password). Off by default; on only on Render.
    demo_login_enabled: bool = False

    # --- Database ---
    database_url: str = (
        "postgresql+psycopg://medvision:medvision_dev@localhost:5432/medvision"
    )

    # --- JWT / auth ---
    jwt_secret_key: str = "change-me-generate-with-openssl-rand-hex-32"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # --- CORS (comma-separated string in .env; use cors_origins_list for the parsed list) ---
    cors_allowed_origins: str = "http://localhost:3000,http://localhost:8000"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]

    # --- Email (verification) ---
    email_backend: str = "console"           # "console" (dev) or "brevo" (prod)
    brevo_api_key: str = ""
    email_sender_address: str = "dev@medvision.local"
    email_sender_name: str = "MedVision AI"
    email_verification_base_url: str = "http://localhost:8000"


    # --- Object storage (MinIO local / R2 prod) ---
    storage_backend: str = "minio"
    s3_endpoint_url: str = "http://localhost:9000"
    s3_bucket: str = "medvision-scans"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"


@lru_cache
def get_settings() -> Settings:
    return Settings()
