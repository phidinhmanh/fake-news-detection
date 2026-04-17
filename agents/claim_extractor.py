"""
claim_extractor.py — Agent 1: Trích xuất Claim chính từ bài viết
==================================================================
Đọc bài viết tiếng Việt → trả về 1-5 claim chính cần kiểm chứng.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Prompt Template ────────────────────────────────────────────────
CLAIM_EXTRACTION_PROMPT = """Bạn là chuyên gia fact-checking tiếng Việt. Nhiệm vụ: phân tích bài viết và trích xuất các luận điểm (claim) chính cần kiểm chứng.

Hãy phân tích bài viết sau và trích xuất 1-5 claim chính.
Mỗi claim phải là một phát biểu cụ thể, có thể kiểm chứng được (factual statement).

Trả về JSON đúng format sau:
{{
    "claims": [
        {{
            "text": "Nội dung claim",
            "importance": 0.9,
            "category": "health/politics/economy/society/technology"
        }}
    ],
    "article_summary": "Tóm tắt ngắn 1-2 câu",
    "overall_credibility_hint": 0.5
}}

--- BÀI VIẾT ---
{article_text}
"""


# ── Schema ─────────────────────────────────────────────────────────
class ExtractedClaim(BaseModel):
    """Một claim được trích xuất."""

    text: str = Field(description="Nội dung claim cần kiểm chứng")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="Mức độ quan trọng")
    category: str = Field(default="general", description="Lĩnh vực: health/politics/economy/society")


class ClaimExtractionResult(BaseModel):
    """Kết quả trích xuất claims."""

    claims: list[ExtractedClaim] = Field(default_factory=list)
    article_summary: str = Field(default="")
    overall_credibility_hint: float = Field(default=0.5, ge=0.0, le=1.0)


# ── Agent ──────────────────────────────────────────────────────────
class ClaimExtractor:
    """Agent 1: Trích xuất claim chính từ bài viết.

    Usage:
        from sequential_adversarial.llm_client import LLMClient
        llm = LLMClient()
        extractor = ClaimExtractor(llm)
        result = extractor.extract("Nội dung bài viết...")
    """

    def __init__(self, llm_client):
        """
        Args:
            llm_client: LLMClient instance (from sequential_adversarial.llm_client).
        """
        self.llm = llm_client

    def extract(self, article_text: str, max_claims: int = 5) -> ClaimExtractionResult:
        """Trích xuất claims từ bài viết.

        Args:
            article_text: Nội dung bài viết.
            max_claims: Số claims tối đa.

        Returns:
            ClaimExtractionResult object.
        """
        if not article_text or len(article_text.strip()) < 20:
            logger.warning("⚠️ [Agent 1] Bài viết quá ngắn.")
            return ClaimExtractionResult(
                claims=[ExtractedClaim(text=article_text[:200], importance=0.5)],
                article_summary="Bài viết quá ngắn để phân tích.",
            )

        prompt = CLAIM_EXTRACTION_PROMPT.format(article_text=article_text[:4000])

        try:
            result = self.llm.generate_structured(
                prompt=prompt,
                schema=ClaimExtractionResult,
                stage_key="claim_extractor",
            )

            # Limit claims
            if len(result.claims) > max_claims:
                result.claims = sorted(result.claims, key=lambda c: c.importance, reverse=True)[:max_claims]

            logger.info(f"🔍 [Agent 1] Extracted {len(result.claims)} claims")
            return result

        except Exception as exc:
            logger.warning(f"⚠️ [Agent 1] LLM error: {exc}. Using fallback.")
            return ClaimExtractionResult(
                claims=[ExtractedClaim(text=article_text[:200], importance=0.5)],
                article_summary=f"Lỗi trích xuất: {str(exc)[:100]}",
            )
