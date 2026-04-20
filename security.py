"""
security.py — Security Utilities for Verity
==========================================
Contains output sanitization and input validation helpers.
Follows NFR-7.2 (Output sanitization) and NFR-7.1 (Input validation).
"""
from __future__ import annotations

import html
import re
import unicodedata
from typing import Literal

import streamlit as st

# Vietnamese diacritics pattern
VIETNAMESE_DIACRITICS = "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"

# Common XSS patterns
XSS_PATTERNS = [
    (r"<script[^>]*>.*?</script>", ""),
    (r"javascript:", ""),
    (r"on\w+\s*=", ""),
    (r"<iframe[^>]*>.*?</iframe>", ""),
    (r"<object[^>]*>.*?</object>", ""),
    (r"<embed[^>]*>", ""),
    (r"data:text/html", ""),
]


def sanitize_for_display(text: str) -> str:
    """
    Sanitize text for safe display in web UI (NFR-7.2).

    Prevents XSS attacks by escaping HTML and removing dangerous patterns.
    """
    if not text:
        return ""

    # Escape HTML entities
    safe_text = html.escape(text)

    # Remove XSS patterns
    for pattern, replacement in XSS_PATTERNS:
        safe_text = re.sub(pattern, replacement, safe_text, flags=re.IGNORECASE | re.DOTALL)

    # Remove null bytes
    safe_text = safe_text.replace("\x00", "")

    # Remove control characters
    safe_text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", safe_text)

    return safe_text


def sanitize_html(text: str) -> str:
    """
    Sanitize HTML content while preserving safe formatting.

    Use st.markdown with this for safe rich text display.
    """
    # Remove all script and style tags
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.IGNORECASE | re.DOTALL)

    # Remove event handlers
    text = re.sub(r"\s*on\w+\s*=\s*[\"'][^\"']*[\"']", "", text, flags=re.IGNORECASE)

    # Remove javascript: URLs
    text = re.sub(r"javascript:", "", text, flags=re.IGNORECASE)

    return text


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode text for consistent processing (VN-2).

    Applies NFC normalization and removes zero-width characters.
    """
    if not text:
        return ""

    # NFC normalization (compose characters)
    text = unicodedata.normalize("NFC", text)

    # Remove combining diacritical marks (zero-width characters)
    text = "".join(
        c for c in text if unicodedata.category(c) != "Mn"
    )

    return text


def contains_vietnamese(text: str) -> bool:
    """Check if text contains Vietnamese diacritics."""
    return any(c in VIETNAMESE_DIACRITICS for c in text.lower())


def validate_text_input(text: str, min_length: int = 10, max_length: int = 10000) -> tuple[bool, str]:
    """
    Validate text input for processing (FR-1.1, FR-1.2).

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Input text is empty"

    if len(text) < min_length:
        return False, f"Input too short (minimum {min_length} characters)"

    if len(text) > max_length:
        return False, f"Input too long (maximum {max_length} characters)"

    # Check for Vietnamese content (at least some diacritics)
    if not contains_vietnamese(text) and not any(c.isascii() for c in text):
        return False, "No recognizable text content"

    return True, ""


def safe_st_markdown(text: str, **kwargs) -> None:
    """
    Safely render markdown in Streamlit with XSS protection (NFR-7.2).

    Use this instead of direct st.markdown() for user-generated content.
    """
    safe_text = sanitize_for_display(text)
    st.markdown(safe_text, **kwargs)


def safe_st_write(text: str) -> None:
    """
    Safely write text in Streamlit with XSS protection (NFR-7.2).
    """
    safe_text = sanitize_for_display(text)
    st.write(safe_text)


class VerdictFormatter:
    """Format verdict results safely for display."""

    @staticmethod
    def format_verdict(verdict: str | None) -> str:
        """Format verdict string safely."""
        if not verdict:
            return "Unknown"
        safe = sanitize_for_display(verdict.upper())
        return safe

    @staticmethod
    def format_confidence(confidence: float | None) -> str:
        """Format confidence as percentage safely."""
        if confidence is None:
            return "0%"
        return f"{confidence:.0%}"

    @staticmethod
    def format_sources(sources: list[dict]) -> list[dict]:
        """Format sources for safe display."""
        safe_sources = []
        for source in sources:
            safe_sources.append({
                "url": sanitize_for_display(source.get("url", "")),
                "stance": sanitize_for_display(source.get("stance", "")),
                "reliability": source.get("reliability", 0.0),
                "excerpt": sanitize_for_display(source.get("excerpt", "")),
            })
        return safe_sources