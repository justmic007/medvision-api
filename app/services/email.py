"""Email sending — pluggable backend (console for dev, Brevo for prod).

The EMAIL_BACKEND setting picks the implementation. In dev the console backend
prints the message (and any verification link) to stdout, so the whole
verification flow is testable without sending real email. In prod the Brevo
backend sends via the Brevo transactional API using BREVO_API_KEY.

Same pluggable-interface discipline as the rest of the project (D-11): callers
depend on EmailSender, not a concrete backend.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.config import get_settings

_settings = get_settings()


class EmailSender(ABC):
    @abstractmethod
    def send(self, to: str, subject: str, html: str) -> None:
        ...


class ConsoleEmailSender(EmailSender):
    """Dev backend: print the email instead of sending it."""

    def send(self, to: str, subject: str, html: str) -> None:
        print("\n" + "=" * 60)
        print(f"[ConsoleEmail] TO: {to}")
        print(f"[ConsoleEmail] SUBJECT: {subject}")
        print(f"[ConsoleEmail] BODY:\n{html}")
        print("=" * 60 + "\n")


class BrevoEmailSender(EmailSender):
    """Prod backend: send via Brevo's transactional email API."""

    def send(self, to: str, subject: str, html: str) -> None:
        import requests

        resp = requests.post(
            "https://api.brevo.com/v3/smtp/email",
            headers={
                "api-key": _settings.brevo_api_key,
                "Content-Type": "application/json",
                "accept": "application/json",
            },
            json={
                "sender": {
                    "email": _settings.email_sender_address,
                    "name": _settings.email_sender_name,
                },
                "to": [{"email": to}],
                "subject": subject,
                "htmlContent": html,
            },
            timeout=10,
        )
        resp.raise_for_status()


def get_email_sender() -> EmailSender:
    """Return the configured email backend."""
    if _settings.email_backend == "brevo":
        return BrevoEmailSender()
    return ConsoleEmailSender()
