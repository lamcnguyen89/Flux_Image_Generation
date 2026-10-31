"""Same-session protection for form submissions and model controls."""

from hmac import compare_digest
from secrets import token_urlsafe

from flask import session


def csrf_token():
    if "generation_csrf" not in session:
        session["generation_csrf"] = token_urlsafe(32)
    return session["generation_csrf"]


def valid_csrf(value):
    expected = session.get("generation_csrf")
    return (isinstance(expected, str) and isinstance(value, str)
            and compare_digest(expected.encode("utf-8"), value.encode("utf-8")))
