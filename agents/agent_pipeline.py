"""
agent_pipeline.py — Orchestrator kết nối 3 Agents
=====================================================
Luồng: Article → Agent 1 (Claims) → Agent 2 (Evidence) → Agent 3 (Score + Explanation)

Usage:
    from agents.agent_pipeline import AgentPipeline
    pipeline = AgentPipeline(mock=True)
    result = pipeline.analyze("Nội dung bài viết...")
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Pipeline Result Schema ──────────────────────────────────────────
class AgentAnalysisResult(BaseModel):
    """Kết quả phân tích hoàn chỉnh từ 3-agent pipeline."""

    # Core results
    fake_score: int = Field(default=50, description="Score tin giả (0-100%)")
    label: str = Field(default="Suspicious", description="Real / Fake / Suspicious")
    confidence: float = Field(default=0.5, description="Độ tự tin")
    explanation: str = Field(default="", description="Giải thích chi tiết")

    # Agent 1 output
    claims: list[dict] = Field(default_factory=list, description="Claims trích xuất")
    article_summary: str = Field(default="")

    # Agent 2 output
    evidence: list[dict] = Field(default_factory=list, description="Evidence tìm được")
    sources_used: list[str] = Field(default_factory=list)

    # Agent 3 output
    reasoning_steps: list[str] = Field(default_factory=list)
    evidence_citations: list[dict] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list)

    # Metadata
    processing_time_seconds: float = Field(default=0.0)
    agents_status: dict = Field(default_factory=dict)


class AgentPipeline:
    """Orchestrator cho 3-Agent pipeline.

    Pipeline flow:
        Article → ClaimExtractor → EvidenceSearcher → ReasoningScorer → Result

    Args:
        llm_client: LLMClient instance (optional, auto-create if None).
        knowledge_base: KnowledgeBase instance (optional, auto-create if None).
        mock: Force mock mode for testing.
        use_wikipedia: Enable Wikipedia VN search.
    """

    def __init__(
        self,
        llm_client=None,
        knowledge_base=None,
        mock: bool = False,
        use_wikipedia: bool = True,
    ):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

        # LLM Client
        if llm_client is None:
            from sequential_adversarial.llm_client import LLMClient
            self.llm = LLMClient(mock=mock)
        else:
            self.llm = llm_client

        # Knowledge Base
        if knowledge_base is None:
            try:
                from agents.knowledge_base import KnowledgeBase
                self.kb = KnowledgeBase()
            except Exception as exc:
                logger.warning(f"⚠️ Knowledge base init failed: {exc}. RAG sẽ bị giới hạn.")
                self.kb = None
        else:
            self.kb = knowledge_base

        # Initialize agents
        from agents.claim_extractor import ClaimExtractor
        from agents.evidence_searcher import EvidenceSearcher
        from agents.reasoning_scorer import ReasoningScorer

        self.claim_extractor = ClaimExtractor(self.llm)
        self.evidence_searcher = EvidenceSearcher(
            knowledge_base=self.kb,
            use_wikipedia=use_wikipedia,
        )
        self.reasoning_scorer = ReasoningScorer(self.llm)

        self.mock = mock
        logger.info(f"🤖 AgentPipeline initialized ({'mock' if mock else 'live'} mode)")

    def analyze(self, article_text: str) -> AgentAnalysisResult:
        """Phân tích bài viết qua 3 agents.

        Args:
            article_text: Nội dung bài viết.

        Returns:
            AgentAnalysisResult object.
        """
        start_time = time.time()
        agents_status: dict = {}

        logger.info("=" * 60)
        logger.info("🚀 [AgentPipeline] BẮT ĐẦU PHÂN TÍCH")
        logger.info("=" * 60)

        # ── Agent 1: Claim Extraction ──────────────────────────────
        try:
            logger.info("\n📌 [Agent 1] Trích xuất claims...")
            claim_result = self.claim_extractor.extract(article_text)
            claims = [c.model_dump() for c in claim_result.claims]
            article_summary = claim_result.article_summary
            agents_status["claim_extractor"] = "success"
        except Exception as exc:
            logger.error(f"❌ Agent 1 failed: {exc}")
            claims = [{"text": article_text[:200], "importance": 0.5, "category": "general"}]
            article_summary = "Lỗi trích xuất claims."
            agents_status["claim_extractor"] = f"error: {str(exc)[:50]}"

        # ── Agent 2: Evidence Search ───────────────────────────────
        try:
            logger.info("\n🔎 [Agent 2] Tìm evidence...")
            claim_texts = [c["text"] for c in claims]
            evidence_result = self.evidence_searcher.search_claims(claim_texts)

            evidence = []
            for ce in evidence_result.claim_evidences:
                for e in ce.evidences:
                    evidence.append(e.model_dump())

            sources_used = evidence_result.sources_used
            agents_status["evidence_searcher"] = "success"
        except Exception as exc:
            logger.error(f"❌ Agent 2 failed: {exc}")
            evidence = []
            sources_used = []
            agents_status["evidence_searcher"] = f"error: {str(exc)[:50]}"

        # ── Agent 3: Reasoning + Scoring ───────────────────────────
        try:
            logger.info("\n🧠 [Agent 3] Reasoning & scoring...")
            reasoning_result = self.reasoning_scorer.reason(
                claims=claims,
                evidence=evidence,
                article_text=article_text,
            )
            agents_status["reasoning_scorer"] = "success"
        except Exception as exc:
            logger.error(f"❌ Agent 3 failed: {exc}")
            from agents.reasoning_scorer import ReasoningResult

            reasoning_result = ReasoningResult(fake_score=50, label="Suspicious", confidence=0.3)
            agents_status["reasoning_scorer"] = f"error: {str(exc)[:50]}"

        # ── Compile Final Result ───────────────────────────────────
        elapsed = time.time() - start_time

        result = AgentAnalysisResult(
            fake_score=reasoning_result.fake_score,
            label=reasoning_result.label,
            confidence=reasoning_result.confidence,
            explanation=reasoning_result.explanation,
            claims=claims,
            article_summary=article_summary,
            evidence=evidence,
            sources_used=sources_used,
            reasoning_steps=reasoning_result.reasoning_steps,
            evidence_citations=[c.model_dump() for c in reasoning_result.evidence_citations],
            risk_factors=reasoning_result.risk_factors,
            processing_time_seconds=round(elapsed, 2),
            agents_status=agents_status,
        )

        logger.info(f"\n{'=' * 60}")
        logger.info(
            f"✅ [AgentPipeline] HOÀN TẤT | "
            f"Score: {result.fake_score}% | "
            f"Label: {result.label} | "
            f"Time: {elapsed:.1f}s"
        )
        logger.info(f"{'=' * 60}")

        return result


def main() -> None:
    """Demo pipeline với mock mode."""
    logging.basicConfig(level=logging.INFO)

    pipeline = AgentPipeline(mock=True)

    test_article = """
    KHẨN CẤP: Vaccine COVID-19 gây ra hàng nghìn ca tử vong trên toàn thế giới! 
    Theo một nghiên cứu bí mật bị rò rỉ, các hãng dược phẩm đã che giấu sự thật 
    về tác dụng phụ nghiêm trọng của vaccine. Chính phủ nhiều nước đã biết nhưng 
    vẫn tiếp tục tiêm chủng cho người dân. Hãy chia sẻ thông tin này ngay!!!
    """

    result = pipeline.analyze(test_article)

    print(f"\n{'=' * 40}")
    print(f"🎯 Fake Score: {result.fake_score}%")
    print(f"🏷️ Label: {result.label}")
    print(f"💪 Confidence: {result.confidence:.2f}")
    print(f"📝 Explanation: {result.explanation}")
    print(f"⏱️ Time: {result.processing_time_seconds}s")
    print(f"{'=' * 40}")


if __name__ == "__main__":
    main()
