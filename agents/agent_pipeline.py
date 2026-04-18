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

        # Khởi tạo lõi xử lí chính (SequentialAdversarialPipeline) thay vì 3 agents rời rạc
        from sequential_adversarial.pipeline import SequentialAdversarialPipeline
        
        self.mock = mock
        # Chúng ta bỏ qua use_wikipedia và knowledge_base manual ở đây vì 
        # SequentialAdversarialPipeline.__init__ đã tự handle việc tích hợp RAG EvidenceSearcher!
        self.pipeline = SequentialAdversarialPipeline(llm=llm_client, mock=self.mock)
        
        logger.info(f"🤖 AgentPipeline initialized ({'mock' if mock else 'live'} mode)")
        logger.info("⚡ Đang sử dụng lõi SequentialAdversarialPipeline (7-Stage Adapter Mode) ⚡")

    def analyze(self, article_text: str) -> AgentAnalysisResult:
        """Phân tích bài viết qua lõi SequentialAdversarial và mapping sang giao diện cũ.

        Args:
            article_text: Nội dung bài viết.

        Returns:
            AgentAnalysisResult object tương thích ngược với web app.
        """
        start_time = time.time()
        agents_status: dict = {}

        logger.info("=" * 60)
        logger.info("🚀 [AgentPipeline Adapter] BẮT ĐẦU PHÂN TÍCH QUA SEQ-PIPELINE")
        logger.info("=" * 60)

        try:
            # 1. Chạy tiến trình 7 bước
            sq_result = self.pipeline.run(article_text)
            
            # 2. Xây dựng Score và Label
            if sq_result.verity_report.conclusion == "False":
                label = "Fake"
                fake_score = int(sq_result.verity_report.confidence * 100)
            elif sq_result.verity_report.conclusion == "True":
                label = "Real"
                fake_score = int((1.0 - sq_result.verity_report.confidence) * 100)
            else:
                label = "Suspicious"
                fake_score = 50

            # 3. Gom nhặt Evidence từ Stage 3 (DataAnalyst -> ClaimAnalysis)
            evidence_list = []
            citations = []
            sources_used = set()
            for ca in sq_result.claim_analyses:
                for src in ca.sources:
                    # Tái cấu trúc source thành format cũ mong đợi của AgentAnalysisResult
                    evidence_list.append({
                        "url": src.url,
                        "title": "Nguồn từ DataAnalyst (Stage 3)",
                        "snippet": src.excerpt
                    })
                    citations.append({
                        "claim": ca.claim.text,
                        "verdict": ca.verdict.upper(),
                        "supporting_evidence": src.excerpt
                    })
                    if src.url:
                        sources_used.add(src.url)

            # Đánh dấu status nội bộ
            agents_status["sq_orchestrator"] = "success"
            
            # 4. Gom nhặt các giải thích (Explanation)
            explanation = (
                f"{sq_result.verity_report.markdown_report}\n\n"
                f"**Ghi chú phản biện (Bias Auditor):**\n{sq_result.bias_report.adversarial_notes}"
            )

            # 5. Khởi tạo đối tượng trả về (Adapter Translation)
            elapsed = time.time() - start_time
            result = AgentAnalysisResult(
                fake_score=fake_score,
                label=label,
                confidence=sq_result.verity_report.confidence,
                explanation=explanation,
                claims=[c.model_dump() for c in sq_result.claims],
                article_summary=sq_result.investigation_summary,
                evidence=evidence_list,
                sources_used=list(sources_used),
                reasoning_steps=sq_result.verity_report.key_findings,
                evidence_citations=citations,
                risk_factors=sq_result.bias_report.analyst_blind_spots,  # mượn field
                processing_time_seconds=round(elapsed, 2),
                agents_status=agents_status,
            )

            logger.info(f"\n{'=' * 60}")
            logger.info(
                f"✅ [AgentPipeline Adapter] HOÀN TẤT | "
                f"Score: {result.fake_score}% | "
                f"Label: {result.label} | "
                f"Time: {elapsed:.1f}s"
            )
            logger.info(f"{'=' * 60}")

            return result

        except Exception as exc:
            logger.error(f"❌ [AgentPipeline Adapter] Thất bại thảm hại: {exc}")
            elapsed = time.time() - start_time
            agents_status["sq_orchestrator"] = f"error: {str(exc)}"
            return AgentAnalysisResult(
                fake_score=50,
                label="Error",
                explanation=f"Có lỗi nghiêm trọng trong lõi xử lý: {exc}",
                processing_time_seconds=round(elapsed, 2),
                agents_status=agents_status
            )


def main() -> None:
    """Demo pipeline với mock mode."""
    logging.basicConfig(level=logging.INFO)

    pipeline = AgentPipeline(mock=False)

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
