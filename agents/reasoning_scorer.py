"""
reasoning_scorer.py — Agent 3: Reasoning + Scoring + Giải thích
=================================================================
Tổng hợp claims + evidence → ra score (0-100%), label, giải thích chi tiết.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Prompt Template ────────────────────────────────────────────────
REASONING_PROMPT = """Bạn là chuyên gia kiểm chứng tin tức (fact-checker) hàng đầu Việt Nam.

Dựa trên các luận điểm (claims) và bằng chứng (evidence) được cung cấp, hãy đưa ra đánh giá cuối cùng.

Hãy suy luận theo từng bước (chain-of-thought) rồi đưa ra kết luận.

Trả về JSON đúng format sau:
{{
    "fake_score": 75,
    "label": "Fake",
    "confidence": 0.85,
    "reasoning_steps": [
        "Bước 1: Phân tích claim chính...",
        "Bước 2: Đối chiếu evidence...",
        "Bước 3: Kết luận..."
    ],
    "explanation": "Giải thích chi tiết 2-3 câu bằng tiếng Việt.",
    "evidence_citations": [
        {{
            "claim": "Claim 1",
            "verdict": "REFUTED",
            "supporting_evidence": "Evidence text..."
        }}
    ],
    "risk_factors": ["Dùng ngôn ngữ gây sốc", "Không có nguồn đáng tin cậy"]
}}

--- CLAIMS ---
{claims_json}

--- EVIDENCE ---
{evidence_json}

--- ORIGINAL ARTICLE (trích đoạn) ---
{article_excerpt}
"""


# ── Schema ─────────────────────────────────────────────────────────
class EvidenceCitation(BaseModel):
    """Trích dẫn evidence cho 1 claim."""

    claim: str = Field(description="Claim gốc")
    verdict: str = Field(default="UNVERIFIED", description="SUPPORTED / REFUTED / MIXED / UNVERIFIED")
    supporting_evidence: str = Field(default="", description="Evidence hỗ trợ")


class ReasoningResult(BaseModel):
    """Kết quả reasoning cuối cùng."""

    fake_score: int = Field(default=50, ge=0, le=100, description="Score tin giả (0-100%)")
    label: str = Field(default="Suspicious", description="Real / Fake / Suspicious")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Độ tự tin")
    reasoning_steps: list[str] = Field(default_factory=list, description="Các bước suy luận")
    explanation: str = Field(default="", description="Giải thích chi tiết tiếng Việt")
    evidence_citations: list[EvidenceCitation] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list, description="Yếu tố rủi ro")


# ── Agent ──────────────────────────────────────────────────────────
class ReasoningScorer:
    """Agent 3: Reasoning + tính score + giải thích.

    Workflow:
        1. Nhận claims + evidence from Agent 1 & 2
        2. Gửi LLM để chain-of-thought reasoning
        3. Trả về: score (0-100%), label, giải thích

    Usage:
        scorer = ReasoningScorer(llm_client)
        result = scorer.reason(claims, evidence, article_text)
    """

    def __init__(self, llm_client):
        """
        Args:
            llm_client: LLMClient instance.
        """
        self.llm = llm_client

    def reason(
        self,
        claims: list[dict],
        evidence: list[dict],
        article_text: str = "",
    ) -> ReasoningResult:
        """Suy luận và tính score.

        Args:
            claims: List of claim dicts (from Agent 1).
            evidence: List of evidence dicts (from Agent 2).
            article_text: Bài viết gốc (trích đoạn).

        Returns:
            ReasoningResult object.
        """
        logger.info(f"🧠 [Agent 3] Reasoning with {len(claims)} claims, {len(evidence)} evidence pieces...")

        claims_json = json.dumps(claims, ensure_ascii=False, indent=2)
        evidence_json = json.dumps(evidence, ensure_ascii=False, indent=2)
        article_excerpt = article_text[:2000] if article_text else "(không có)"

        prompt = REASONING_PROMPT.format(
            claims_json=claims_json,
            evidence_json=evidence_json,
            article_excerpt=article_excerpt,
        )

        try:
            result = self.llm.generate_structured(
                prompt=prompt,
                schema=ReasoningResult,
                stage_key="reasoning_scorer",
            )

            # Validate & normalize
            result = self._normalize_result(result)

            logger.info(
                f"✅ [Agent 3] Score: {result.fake_score}% | "
                f"Label: {result.label} | "
                f"Confidence: {result.confidence:.2f}"
            )
            return result

        except Exception as exc:
            logger.warning(f"⚠️ [Agent 3] LLM error: {exc}. Using heuristic fallback.")
            return self._heuristic_fallback(claims, evidence)

    def _normalize_result(self, result: ReasoningResult) -> ReasoningResult:
        """Normalize & validate kết quả.

        Đảm bảo label khớp với score, confidence hợp lệ.
        """
        # Normalize label based on score
        if result.fake_score >= 70:
            result.label = "Fake"
        elif result.fake_score <= 30:
            result.label = "Real"
        else:
            result.label = "Suspicious"

        # Clamp values
        result.fake_score = max(0, min(100, result.fake_score))
        result.confidence = max(0.0, min(1.0, result.confidence))

        return result

    def _heuristic_fallback(
        self,
        claims: list[dict],
        evidence: list[dict],
    ) -> ReasoningResult:
        """Fallback heuristic khi LLM fails.

        Tính score dựa trên:
        - Số evidence tìm được
        - Stance distribution (support vs refute)
        """
        support_count = 0
        refute_count = 0
        total = len(evidence) if evidence else 1

        for e in evidence:
            stance = e.get("stance", "neutral")
            if stance == "support":
                support_count += 1
            elif stance == "refute":
                refute_count += 1

        # Simple heuristic score
        if total > 0:
            support_ratio = support_count / total
            refute_ratio = refute_count / total

            # Higher refute → more likely fake
            fake_score = int(refute_ratio * 80 + (1 - support_ratio) * 20)
        else:
            fake_score = 50  # No evidence → uncertain

        fake_score = max(0, min(100, fake_score))

        if fake_score >= 70:
            label = "Fake"
        elif fake_score <= 30:
            label = "Real"
        else:
            label = "Suspicious"

        return ReasoningResult(
            fake_score=fake_score,
            label=label,
            confidence=0.4,  # Low confidence for heuristic
            reasoning_steps=[
                f"Fallback heuristic: {support_count} support, {refute_count} refute, {total} total",
                f"Score = refute_ratio * 80 + (1 - support_ratio) * 20 = {fake_score}",
            ],
            explanation=(
                f"Phân tích heuristic (LLM không khả dụng): "
                f"Tìm thấy {support_count} bằng chứng ủng hộ và {refute_count} bằng chứng phản bác."
            ),
            risk_factors=["LLM reasoning không khả dụng, dùng heuristic fallback"],
        )
