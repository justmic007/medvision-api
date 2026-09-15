"""Security primitives: password hashing.

Uses passlib's bcrypt. Passwords are never stored or compared in plaintext —
hash_password() stores a salted bcrypt hash, and verify_password() does a
constant-time comparison (bcrypt handles the salt and timing internally).
"""
from __future__ import annotations

from passlib.context import CryptContext

# bcrypt with passlib's default rounds; deprecated="auto" lets us upgrade
# the scheme later without breaking existing hashes.
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plain: str) -> str:
    """Return a salted bcrypt hash of the password."""
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """Check a plaintext password against a stored hash (constant-time)."""
    return _pwd_context.verify(plain, hashed)
