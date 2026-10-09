"""Request/response schemas for the auth endpoints."""
from pydantic import BaseModel, EmailStr, Field, field_validator


# Password policy: 8-72 chars (72 is bcrypt's hard limit — longer is silently
# truncated, so we reject it rather than mislead). Require a mix so trivial
# passwords ("password", "12345678") are discouraged.
PASSWORD_MIN = 8
PASSWORD_MAX = 72


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=PASSWORD_MIN, max_length=PASSWORD_MAX)

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        has_letter = any(c.isalpha() for c in v)
        has_digit = any(c.isdigit() for c in v)
        if not (has_letter and has_digit):
            raise ValueError(
                "Password must contain at least one letter and one number."
            )
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    # Login does NOT enforce strength — it just checks against the stored hash.
    # (Enforcing here would leak the policy and reject legacy passwords.)
    password: str = Field(min_length=1, max_length=PASSWORD_MAX)


class DemoLoginRequest(BaseModel):
    """Public demo: the browser sends ONLY the role. The backend maps it to a
    fixed demo-domain account; no email or password ever crosses the wire."""
    role: str

    @field_validator("role")
    @classmethod
    def valid_role(cls, v: str) -> str:
        if v not in {"clinician", "admin"}:
            raise ValueError("role must be 'clinician' or 'admin'")
        return v


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class AccessTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class MessageResponse(BaseModel):
    message: str


class UserResponse(BaseModel):
    id: str
    email: str
    role: str
    status: str
    email_verified: bool
