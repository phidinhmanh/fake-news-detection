"""
input_processor.py — Stage 1: Input & Trigger
==============================================
Auto-detects input type (URL / file path / raw text) and extracts plain text.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


class InputProcessor:
    """
    Stage 1: Input & Trigger.

    Accepts:
    - A URL (http/https) → fetches page and extracts visible text
    - A file path → reads .txt or .pdf files
    - Raw text string → passes through directly

    Returns an InputData-compatible dict with keys:
      source, raw_text, input_type, metadata
    """

    # Maximum characters we pass forward (avoids giant context windows)
    MAX_CHARS = 8_000

    def process(self, source: str) -> dict:
        source = source.strip()
        input_type = self._detect_type(source)

        if input_type == "url":
            raw_text, metadata = self._fetch_url(source)
        elif input_type == "file":
            raw_text, metadata = self._read_file(source)
        else:
            raw_text, metadata = self._handle_raw(source)

        raw_text = raw_text[: self.MAX_CHARS]

        return {
            "source": source,
            "raw_text": raw_text,
            "input_type": input_type,
            "metadata": metadata,
        }

    # ── Type detection ─────────────────────────────────────────────────────────
    def _detect_type(self, source: str) -> str:
        if re.match(r"^https?://", source, re.IGNORECASE):
            return "url"
        try:
            if Path(source).is_file():
                return "file"
        except Exception:
            pass
        return "raw"

    # ── URL handler ────────────────────────────────────────────────────────────
    def _fetch_url(self, url: str) -> tuple[str, dict]:
        """Fetch URL and extract readable text."""
        try:
            # Try trafilatura first (best quality extraction)
            import trafilatura  # type: ignore

            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded) or ""
                metadata = trafilatura.extract_metadata(downloaded)
                meta_dict = {}
                if metadata:
                    meta_dict = {
                        "title": getattr(metadata, "title", ""),
                        "author": getattr(metadata, "author", ""),
                        "date": getattr(metadata, "date", ""),
                        "domain": getattr(metadata, "hostname", ""),
                    }
                return text, {**meta_dict, "fetcher": "trafilatura"}
        except ImportError:
            pass  # trafilatura not installed — fall back to httpx

        try:
            import httpx  # type: ignore
            from html.parser import HTMLParser

            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
            response = httpx.get(url, timeout=10, follow_redirects=True, headers=headers)
            response.raise_for_status()
            text = self._strip_html(response.text)
            domain = re.sub(r"https?://(www\.)?([^/]+).*", r"\2", url)
            return text, {"domain": domain, "fetcher": "httpx"}
        except Exception as exc:
            return f"[URL fetch failed: {exc}]", {"error": str(exc)}

    @staticmethod
    def _strip_html(html: str) -> str:
        """Very basic HTML → plain text conversion using stdlib."""
        from html.parser import HTMLParser

        class _Stripper(HTMLParser):
            def __init__(self):
                super().__init__()
                self.parts: list[str] = []
                self._skip = False

            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style", "nav", "header", "footer"):
                    self._skip = True

            def handle_endtag(self, tag):
                if tag in ("script", "style", "nav", "header", "footer"):
                    self._skip = False

            def handle_data(self, data):
                if not self._skip:
                    stripped = data.strip()
                    if stripped:
                        self.parts.append(stripped)

        stripper = _Stripper()
        stripper.feed(html)
        return " ".join(stripper.parts)

    # ── File handler ────────────────────────────────────────────────────────────
    def _read_file(self, path: str) -> tuple[str, dict]:
        """Read a text or PDF file."""
        p = Path(path)
        suffix = p.suffix.lower()
        metadata = {"filename": p.name, "suffix": suffix}

        if suffix == ".pdf":
            return self._read_pdf(p), metadata
        else:
            # Treat everything else as plain text (utf-8 with fallback)
            try:
                text = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = p.read_text(encoding="latin-1")
            return text, metadata

    @staticmethod
    def _read_pdf(path: Path) -> str:
        """Extract text from PDF using pypdf or pdfplumber (if available)."""
        try:
            import pypdf  # type: ignore

            reader = pypdf.PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)
        except ImportError:
            pass

        try:
            import pdfplumber  # type: ignore

            with pdfplumber.open(str(path)) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
            return "\n".join(pages)
        except ImportError:
            return f"[PDF extraction requires pypdf or pdfplumber: pip install pypdf]"

    # ── Raw text handler ───────────────────────────────────────────────────────
    @staticmethod
    def _handle_raw(text: str) -> tuple[str, dict]:
        return text, {"source_type": "raw_text", "char_count": len(text)}
