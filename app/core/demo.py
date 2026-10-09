"""Backend-only constants for the public demo.

These never reach the frontend: the browser sends only {role} to
/auth/demo-login, and the backend maps that to a fixed demo-domain email
here. Everything demo-related (demo-login, read-only enforcement, the seed,
the per-login reset, admin read-scoping) keys off these values and the
users.is_demo flag, so real accounts are never affected.
"""
from __future__ import annotations

DEMO_DOMAIN = "demo.medvision.dev"

DEMO_ACCOUNTS: dict[str, str] = {
    "clinician": f"demo-clinician@{DEMO_DOMAIN}",
    "admin": f"demo-admin@{DEMO_DOMAIN}",
}

DEMO_SACRIFICIAL_CLINICIAN = f"demo-clinician-b@{DEMO_DOMAIN}"

DEMO_SACRIFICIAL_CANONICAL_STATUS = "approved"


def is_demo_email(email: str) -> bool:
    """True if the email is on the demo domain."""
    return email.lower().endswith("@" + DEMO_DOMAIN)
