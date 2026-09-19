"""Object storage — scan images (MinIO local / Cloudflare R2 prod).

Both MinIO and R2 are S3-compatible, so one boto3 client serves both; only the
endpoint and credentials differ (from config). Scans are stored under a key and
referenced from the case (D-10) — the pixels live here, never in Postgres.
"""
from __future__ import annotations

import uuid

import boto3
from botocore.client import Config

from app.core.config import get_settings

_settings = get_settings()


def _client():
    return boto3.client(
        "s3",
        endpoint_url=_settings.s3_endpoint_url,
        aws_access_key_id=_settings.s3_access_key,
        aws_secret_access_key=_settings.s3_secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def ensure_bucket() -> None:
    """Create the bucket if it doesn't exist (idempotent)."""
    s3 = _client()
    existing = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
    if _settings.s3_bucket not in existing:
        s3.create_bucket(Bucket=_settings.s3_bucket)


def upload_scan(data: bytes, suffix: str) -> str:
    """Store scan bytes under a new key; return the key."""
    key = f"scans/{uuid.uuid4()}{suffix}"
    _client().put_object(Bucket=_settings.s3_bucket, Key=key, Body=data)
    return key


def download_scan(key: str) -> bytes:
    """Fetch scan bytes by key."""
    obj = _client().get_object(Bucket=_settings.s3_bucket, Key=key)
    return obj["Body"].read()
