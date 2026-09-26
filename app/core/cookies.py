"""Refresh-token cookie handling for the httpOnly-cookie auth pattern.

The refresh token is stored in an httpOnly cookie so browser JavaScript can
never read it (XSS can't steal it). Cookie attributes differ by environment:
  - dev  (same-site localhost): SameSite=Lax, not Secure (http is fine locally)
  - prod (cross-site Vercel->Render): SameSite=None, Secure (needs HTTPS both ends)
Controlled by ENVIRONMENT so the same code works in both.
"""
from __future__ import annotations

from fastapi import Response

from app.core.config import get_settings

REFRESH_COOKIE_NAME = "refresh_token"


def set_refresh_cookie(response: Response, token: str) -> None:
    settings = get_settings()
    is_prod = settings.environment == "production"
    max_age = settings.refresh_token_expire_days * 24 * 60 * 60
    response.set_cookie(
        key=REFRESH_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=is_prod,                       # HTTPS-only in prod
        samesite="none" if is_prod else "lax",  # cross-site in prod, same-site in dev
        max_age=max_age,
        path="/auth",                         # only sent to /auth/* endpoints
    )


def clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(key=REFRESH_COOKIE_NAME, path="/auth")
