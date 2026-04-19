"""
pipeline.py — Sequential Adversarial Pipeline (8 Stages)
================================================================================
Tổ chức code theo Design Pattern: "Chain of Responsibility" (Chuỗi trách nhiệm).
Mỗi bước (stage) lặp lại một interface cố định: process(result) -> result.
Dữ liệu đầu ra của hệ thống được gói gọn trong class PipelineResult.

Luồng đi (Workflow):
  1. InputProcessor   — Trích xuất văn bản từ URL/File.
  2. LeadInvestigator — (LLM) Tìm Luận điểm (Claims) & Từ ngữ thao túng (Loaded Language).
  3. DataAnalyst      — (LLM) Đóng vai thư viện số, kiểm duyệt/đối chứng các Luận điểm.
  4. BiasAuditor      — (LLM) Phản biện Data Analyst để tìm Định kiến (Bias) và góc khuất.
  5. Synthesizer      — (LLM) Tổng hợp báo cáo cuối cùng (Verity Report) dựa trên bối cảnh.
  6. VisualEngine     — Vẽ sơ đồ Dòng chảy logic bằng Mermaid.js.
  7. Persistence      — Lưu log và cấu trúc Json vào Sqlite Database.
  8. TFIDFComparator  — Đối chiếu kết quả AI với model Baseline truyền thống (Hệ An Toàn).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

# Tools & LLM Clients
from sequential_adversarial.input_processor import InputProcessor
from sequential_adversarial.llm_client import BaseLLMProvider, LLMClientFactory

# Contract Models (Pydantic schemas giúp ép Schema Output cho LLM và Auto-complete IDE)
from sequential_adversarial.models import (
    Claim, ClaimAnalysis, InvestigationResult, AnalysisResult,
    BiasReport, VerityReport, PipelineResult, TFIDFComparison
)
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

@dataclass
class PipelineResources:
    """Container for shared resources to avoid redundant loading."""
    llm: BaseLLMProvider
    searcher: Optional[Any] = None
    tfidf_model: Optional[Any] = None
    input_processor: Optional[InputProcessor] = None

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────────
# [Stage 2] Lead Investigator: Kẻ hoài nghi
# ───────────────────────────────────────────────────────────────────────────────
class LeadInvestigator:
    """Đọc lướt văn bản để trích xuất Luận điểm cốt lõi và phát hiện từ ngữ thao túng."""
    
    PROMPT_TEMPLATE = """You are a lead investigative journalist specialising in fact-checking.

Analyse the following article text and extract 2-4 core factual claims worth investigating.
For each claim, score its suspicion (0.0=benign, 1.0=very suspicious) and list loaded/manipulative words.

Respond ONLY with valid JSON exactly matching the requested schema.

--- ARTICLE TEXT ---
{text}
"""

    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm

    def process(self, result: PipelineResult) -> PipelineResult:
        logger.info("🔍 [Stage 2] LeadInvestigator: Đang tìm kiếm luận điểm cốt lõi...")
        prompt = self.PROMPT_TEMPLATE.format(text=result.raw_text[:4000])

        try:
            # Sức mạnh lớn nhất: Ép kiểu LLM trả về đúng 100% chuẩn InvestigationResult bằng Pydantic Model
            inv = self.llm.generate_structured(
                prompt=prompt, 
                schema=InvestigationResult, 
                stage_key="lead_investigator"
            )
        except Exception as exc:
            logger.warning(f"⚠️ [Stage 2] Lỗi LLM: {exc}. Dùng Fallback (Mọi thứ ko bao giờ bị sập cứng).")
            inv = InvestigationResult(
                claims=[Claim(text=result.raw_text[:200], suspicion_score=0.5, loaded_language=[])],
                overall_manipulation_score=0.5,
                summary=f"LỖI TRÍCH XUẤT (Fallback Triggered): {str(exc)}",
            )

        # Cập nhật Result object và chuyển cho Stage tiếp theo theo phong cách Chain-of-responsibility
        result.claims = inv.claims
        result.overall_manipulation_score = inv.overall_manipulation_score
        result.investigation_summary = inv.summary
        
        return result


# ───────────────────────────────────────────────────────────────────────────────
# [Stage 3] Data Analyst: Máy Quét Đối Chứng (RAG)
# ───────────────────────────────────────────────────────────────────────────────
class DataAnalyst:
    """Sử dụng EvidenceSearcher để quét Wikipedia/DB và dùng LLM so khớp Bằng Chứng thật."""

    PROMPT_TEMPLATE = """You are a senior data analyst at a fact-checking organisation.

Evaluate the provided claims based ONLY on the attached REAL evidence gathered from our Knowledge Base and Wikipedia. 
For EACH claim, list the provided evidence sources as your `sources` list and determine a verdict ("supported", "refuted", "mixed", or "unverified").
Do NOT invent hypothetical sources. Use only what is provided in the gathered evidence. If no evidence is provided, mark as unverified.

Respond ONLY with valid JSON exactly matching the requested schema.

--- CLAIMS AND GATHERED REAL EVIDENCE ---
{payload_json}
"""

    def __init__(self, llm: BaseLLMProvider, evidence_searcher=None):
        self.llm = llm
        self.searcher = evidence_searcher

    def process(self, result: PipelineResult) -> PipelineResult:
        logger.info("📚 [Stage 3] DataAnalyst: Đang thu thập và đối chiếu Bằng Chứng Thực Tế...")
        
        # Optimize: Parallel RAG Search
        if not result.claims:
            return result
            
        def _fetch_evidence(claim):
            evidence_data = []
            if self.searcher:
                try:
                    ce = self.searcher.search_single_claim(claim.text)
                    evidence_data = [e.model_dump() for e in ce.evidences]
                except Exception as exc:
                    logger.warning(f"⚠️ Không thể RAG cho claim '{claim.text[:30]}...': {exc}")
            return {
                "claim": claim.model_dump(),
                "gathered_real_evidence": evidence_data
            }

        logger.info(f"⚡ [Stage 3] Đang thực hiện search song song cho {len(result.claims)} claims...")
        with ThreadPoolExecutor() as executor:
            payload = list(executor.map(_fetch_evidence, result.claims))
            
        payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
        prompt = self.PROMPT_TEMPLATE.format(payload_json=payload_json)

        try:
            analysis = self.llm.generate_structured(
                prompt=prompt, 
                schema=AnalysisResult, 
                stage_key="data_analyst"
            )
        except Exception as exc:
            logger.warning(f"⚠️ [Stage 3] Lỗi LLM: {exc}. Dùng Fallback.")
            analysis = AnalysisResult(
                claim_analyses=[
                    ClaimAnalysis(claim=claim, sources=[], verdict="unverified")
                    for claim in result.claims
                ],
                summary=f"LỖI ĐỐI SOÁT (Fallback Triggered): {str(exc)}",
            )

        result.claim_analyses = analysis.claim_analyses
        result.analysis_summary = analysis.summary
        return result


# ───────────────────────────────────────────────────────────────────────────────
# [Stage 4] Bias Auditor: Kẻ phản biện (Adversarial Step)
# ───────────────────────────────────────────────────────────────────────────────
class BiasAuditor:
    """Bắt bẻ lại lý lẽ của Data Analyst để tìm kiếm định kiến, Framing và thiếu sót logic."""

    PROMPT_TEMPLATE = """You are an adversarial bias auditor challenging the previous analyst's findings.
Identify framing bias, narrative distortion, and unexamined assumptions.

Respond ONLY with valid JSON exactly matching the requested schema.

--- ARTICLE EXCERPT ---
{text}

--- ANALYST FINDINGS ---
{analysis_json}
"""

    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm

    def process(self, result: PipelineResult) -> PipelineResult:
        logger.info("⚖️ [Stage 4] BiasAuditor: Đang tìm kiếm định kiến và phản biện...")
        
        analysis_json = json.dumps([ca.model_dump() for ca in result.claim_analyses], ensure_ascii=False)
        prompt = self.PROMPT_TEMPLATE.format(
            text=result.raw_text[:2000],
            analysis_json=analysis_json,
        )

        try:
            bias = self.llm.generate_structured(
                prompt=prompt, 
                schema=BiasReport, 
                stage_key="bias_auditor"
            )
        except Exception as exc:
            logger.warning(f"⚠️ [Stage 4] Lỗi LLM: {exc}. Dùng Fallback.")
            bias = BiasReport(
                framing="Unknown / Lỗi hệ thống",
                distortion_detected=False,
                adversarial_notes=f"LỖI PHẢN BIỆN (Fallback Triggered): {str(exc)}",
            )

        result.bias_report = bias
        return result


# ───────────────────────────────────────────────────────────────────────────────
# [Stage 5] Synthesizer: Người đúc kết
# ───────────────────────────────────────────────────────────────────────────────
class Synthesizer:
    """Tổng hợp toàn bộ thông tin các LLM Agent ở trên thành 1 Verity Report hoàn chỉnh."""

    PROMPT_TEMPLATE = """You are the Chief Verification Officer. Based on all evidence collected, 
produce a final Verity Report (True, False, or Mixed).

Respond ONLY with valid JSON exactly matching the requested schema.

--- INVESTIGATION DATA ---
Claims: {claims_json}
Analysis: {analysis_json}
Bias Report: {bias_json}
"""

    def __init__(self, llm: BaseLLMProvider):
        self.llm = llm

    def process(self, result: PipelineResult) -> PipelineResult:
        logger.info("📝 [Stage 5] Synthesizer: Đang viết báo cáo cuối cùng (Verity Report)...")
        
        claims_json = json.dumps([c.model_dump() for c in result.claims], ensure_ascii=False)
        analysis_json = json.dumps([ca.model_dump() for ca in result.claim_analyses], ensure_ascii=False)
        bias_json = result.bias_report.model_dump_json() if result.bias_report else "{}"

        prompt = self.PROMPT_TEMPLATE.format(
            claims_json=claims_json,
            analysis_json=analysis_json,
            bias_json=bias_json,
        )

        try:
            verity = self.llm.generate_structured(
                prompt=prompt, 
                schema=VerityReport, 
                stage_key="synthesizer"
            )
        except Exception as exc:
            logger.warning(f"⚠️ [Stage 5] Lỗi LLM: {exc}. Dùng Fallback.")
            verity = VerityReport(
                conclusion="Mixed",
                confidence=0.5,
                evidence_summary=f"LỖI TỔNG HỢP (Fallback Triggered): {str(exc)}",
                markdown_report=f"## Lỗi Hệ Thống\nKhông thể tạo Verity Report do lỗi: {str(exc)}",
            )

        result.verity_report = verity
        return result


# ───────────────────────────────────────────────────────────────────────────────
# [Stage 6] Visual Engine: Vẽ sơ đồ luồng dữ liệu (Visual Flow)
# ───────────────────────────────────────────────────────────────────────────────
class VisualEngine:
    """Trực quan hóa Dòng chảy Logic của bằng chứng qua Mermaid.js / File Markdown."""

    def __init__(self, output_dir: Optional[Path] = None):
        from config import SA_VISUAL_DIR
        self.output_dir = output_dir or SA_VISUAL_DIR

    def process(self, result: PipelineResult) -> PipelineResult:
        logger.info("📊 [Stage 6] VisualEngine: Đang vẽ biểu đồ logic Mermaid...")
        
        diagram = self._build_diagram(result)
        result.mermaid_diagram = diagram

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = self.output_dir / f"verity_flowchart_{ts}.md"
            md_content = f"# Báo cáo Xác thực — Dòng chảy Logic\n\n```mermaid\n{diagram}\n```\n"
            out_path.write_text(md_content, encoding="utf-8")
            
            result.visual_flowchart_path = str(out_path)
            logger.info(f"💾 Biểu đồ logic đã được xuất tại: {out_path}")
        except Exception as exc:
            logger.warning(f"⚠️ [Stage 6] Không thể lưu file .md đồ thị: {exc}")

        return result

    def _build_diagram(self, result: PipelineResult) -> str:
        """Tự động gen Node logic dựa trên Verdict của từng step."""
        lines = ["flowchart TD"]
        lines.append(f'    IN["📰 Article Input\\n{self._truncate(result.source, 40)}"]')
        lines.append('    IN --> INV["🔍 Lead Investigator"]')

        for i, claim in enumerate(result.claims[:3]):
            cid = f"C{i}"
            label = self._truncate(claim.text, 35)
            score = f"{claim.suspicion_score:.0%}"
            lines.append(f'    INV --> {cid}["⚠️ Claim {i+1}: {label}\\nSuspicion: {score}"]')

            if i < len(result.claim_analyses):
                verdict = result.claim_analyses[i].verdict.upper()
                emoji = {"REFUTED": "❌", "SUPPORTED": "✅", "MIXED": "🟡", "UNVERIFIED": "❓"}.get(verdict, "❓")
                lines.append(f'    {cid} --> V{i}["{emoji} {verdict}"]')

        if result.bias_report:
            distortion = "⚠️ Bias Detected" if result.bias_report.distortion_detected else "✅ No Bias"
            framing = self._truncate(result.bias_report.framing, 30)
            lines.append(f'    INV --> BIAS["🧐 Bias Auditor\\n{distortion}\\nFraming: {framing}"]')

        if result.verity_report:
            conclusion = result.verity_report.conclusion
            emoji = {"True": "✅", "False": "❌", "Mixed": "🟡"}.get(conclusion, "❓")
            confidence = f"{result.verity_report.confidence:.0%}"
            lines.append(f'    BIAS --> FINAL["{emoji} VERDICT: {conclusion}\\nConfidence: {confidence}"]')
            lines.append(f'    FINAL --> DB["💾 Lưu Log SQLite Database"]')

        return "\n".join(lines)

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        text = str(text).replace('"', "'").replace("\n", " ")
        return text[:max_len] + "..." if len(text) > max_len else text


# ───────────────────────────────────────────────────────────────────────────────
# [Stage 7] Persistence: Lưu trữ Database
# ───────────────────────────────────────────────────────────────────────────────
class Persistence:
    """Lưu trữ Log toàn bộ cấu trúc Pipeline Json siêu bự vào SQL để tiện truy xuất."""

    def __init__(self, db_path: Optional[Path] = None):
        from config import SA_DB_PATH
        self.db_path = db_path or SA_DB_PATH
        self._ensure_db()

    def _ensure_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    source      TEXT    NOT NULL,
                    conclusion  TEXT    NOT NULL,
                    confidence  REAL    NOT NULL DEFAULT 0.5,
                    manipulation_score REAL NOT NULL DEFAULT 0.0,
                    full_json   TEXT    NOT NULL,
                    created_at  TEXT    NOT NULL
                )
            """)
            conn.commit()

    def process(self, result: PipelineResult) -> PipelineResult:
        logger.info("💽 [Stage 7] Persistence: Đang lưu object dữ liệu ngầm vào SQLite Database...")
        
        conclusion = result.verity_report.conclusion if result.verity_report else "Unknown"
        confidence = result.verity_report.confidence if result.verity_report else 0.0
        created_at = datetime.now().isoformat()
        full_json = result.model_dump_json(indent=None)  # Dump an toàn Pydantic Schema.

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """INSERT INTO reports
                       (source, conclusion, confidence, manipulation_score, full_json, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (result.source, conclusion, confidence, result.overall_manipulation_score, full_json, created_at),
                )
                record_id = cursor.lastrowid
                conn.commit()

            result.db_record_id = record_id
            result.saved_id = f"db_{record_id}"
        except Exception as exc:
            logger.error(f"⚠️ [Stage 7] Không thể lưu Database, lỗi I/O System: {exc}")

        return result

    def fetch_by_id(self, record_id: int) -> Optional[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM reports WHERE id = ?", (record_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def fetch_all(self) -> list[dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM reports ORDER BY id DESC")
            return [dict(row) for row in cursor.fetchall()]


# ───────────────────────────────────────────────────────────────────────────────
# [Stage 8] TF-IDF Comparator: Lớp học Kẽ Đá Toán Học Cũ (Baseline)
# ───────────────────────────────────────────────────────────────────────────────
class TFIDFComparator:
    """
    So sánh kết luận của Hệ đa môi giới LLM với kết quả Mô hình TF-IDF cổ điển.
    Mục đích: Nếu LLM ảo giác (Hallucination), model cơ bản này có thể dấy lên hồi chuông cảnh báo.
    """

    def __init__(self, mock: bool = False, model=None):
        self.mock = mock
        self._model = model

    def _load_model(self):
        """Lazy load model: Nghĩa là chỉ truy xuất HDD lấy mô hình khi Stage 8 thực sự chạy."""
        if self._model is None and not self.mock:
            try:
                from model.baseline_logreg import BaselineLogReg
                model = BaselineLogReg()
                model.load()
                self._model = model
                logger.info("⚡ [Stage 8] TF-IDF model loaded successfully.")
            except Exception as exc:
                logger.warning(f"⚠️ [Stage 8] Nạp model cổ điển thất bại, file pkl chưa có? - {exc}")
                self._model = None

    def process(self, result: PipelineResult) -> PipelineResult:
        logger.info("🤖 [Stage 8] TF-IDF Comparator: Đối chiếu lại quyết định LLM và Model toán học...")

        if self.mock:
            tfidf_label, fake_proba, real_proba = "fake", 0.72, 0.28
        else:
            self._load_model()
            if self._model is None:
                return result

            try:
                scores = self._model.predict_with_score(result.raw_text)
                tfidf_label = scores["label"]
                fake_proba, real_proba = scores["fake_proba"], scores["real_proba"]
            except Exception as exc:
                logger.warning(f"⚠️ [Stage 8] Phân tích điểm xác suất NLP thất bại: {exc}")
                return result

        """
        Khớp Quyết định giữa 2 Mô hình:
         - LLM phán: 'False'/ 'Mixed' => Cùng phe 'Fake'
         - TF-IDF phán: 'fake'
        """
        llm_verdict = result.verity_report.conclusion if result.verity_report else None
        llm_confidence = result.verity_report.confidence if result.verity_report else 0.0

        llm_predicted_fake = (llm_verdict in ("False", "Mixed"))
        tfidf_predicted_fake = (tfidf_label == "fake")
        agreement = (llm_predicted_fake == tfidf_predicted_fake)

        notes = (
            "✅ HAI MÔ HÌNH THỐNG NHẤT: Độ tin cậy siêu cao!" if agreement else
            f"❌ XUNG ĐỘT KÉP: LLM bảo '{llm_verdict}', nhưng TF-IDF lại kêu '{tfidf_label}' "
            f"(với P_Fake={fake_proba:.2f}). Lệnh cho Biên Tập Viên con người vào duyệt!"
        )

        result.tfidf_comparison = TFIDFComparison(
            tfidf_label=tfidf_label,
            tfidf_confidence=max(fake_proba, real_proba),
            fake_proba=fake_proba,
            real_proba=real_proba,
            llm_verdict=llm_verdict,
            llm_confidence=llm_confidence,
            agreement=agreement,
            disagreement_notes=notes,
        )
        return result


# ───────────────────────────────────────────────────────────────────────────────
# Bộ Điều Phối Chính (The Main Orchestrator)
# ───────────────────────────────────────────────────────────────────────────────
class SequentialAdversarialPipeline:
    """
    Điểm đón dữ liệu duy nhất (Entry-point). 
    Viết luồng giống như chơi trượt ống nước: Đầu vào là Source, nhét ống chạy tự động -> Đầu ra là Result.

    Cách sử dụng:
        pipeline = SequentialAdversarialPipeline(mock=True)
        result = pipeline.run("https://news.com/fake-article")
        print(result.verity_report.conclusion)
    """

    def __init__(self, llm: Optional[BaseLLMProvider] = None, mock: bool = False, resources: Optional[PipelineResources] = None):
        if resources:
            self.llm = resources.llm
            self.input_processor = resources.input_processor or InputProcessor()
            searcher = resources.searcher
            tfidf_model = resources.tfidf_model
        else:
            if llm is None:
                from config import LLM_PROVIDER
                self.llm = LLMClientFactory.create(LLM_PROVIDER, mock=mock)
            else:
                self.llm = llm
            self.input_processor = InputProcessor()
            
            # Khởi tạo Evidence Searcher (RAG System)
            searcher = None
            tfidf_model = None
            if not mock:
                try:
                    from agents.knowledge_base import KnowledgeBase
                    from agents.evidence_searcher import EvidenceSearcher
                    kb = KnowledgeBase()
                    searcher = EvidenceSearcher(knowledge_base=kb)
                    logger.info("✅ Tích hợp EvidenceSearcher RAG thành công.")
                except Exception as exc:
                    logger.warning(f"⚠️ Khởi tạo RAG thất bại: {exc}. Stage 3 sẽ không có RAG.")
        
        # Array này chứa List TẤT CẢ các bước, được pass con trỏ `result` theo dòng chảy
        self.stages = [
            LeadInvestigator(self.llm),
            DataAnalyst(self.llm, evidence_searcher=searcher),
            BiasAuditor(self.llm),
            Synthesizer(self.llm),
            VisualEngine(),
            Persistence(),
            TFIDFComparator(mock=mock, model=tfidf_model),
        ]

    @classmethod
    def load_resources(cls, provider_name: Optional[str] = None, mock: bool = False) -> PipelineResources:
        """Pre-load all heavy resources for the pipeline."""
        logger.info("🏗️  [Pipeline] Đang khởi tạo tài nguyên hệ thống (Pre-loading)...")
        
        # 1. LLM Client
        if provider_name is None:
            from config import LLM_PROVIDER
            provider_name = LLM_PROVIDER
        llm = LLMClientFactory.create(provider_name, mock=mock)
        
        # 2. Input Processor
        input_processor = InputProcessor()
        
        # 3. RAG Searcher
        searcher = None
        if not mock:
            try:
                from agents.knowledge_base import KnowledgeBase
                from agents.evidence_searcher import EvidenceSearcher
                kb = KnowledgeBase()
                # Warm up KnowledgeBase (load encoder)
                kb._get_encoder()
                searcher = EvidenceSearcher(knowledge_base=kb)
                logger.info("✅ Tải RAG Searcher thành công.")
            except Exception as exc:
                logger.warning(f"⚠️ Không thể tải RAG Searcher: {exc}")

        # 4. TF-IDF Model
        tfidf_model = None
        if not mock:
            try:
                from model.baseline_logreg import BaselineLogReg
                tfidf_model = BaselineLogReg()
                tfidf_model.load()
                logger.info("✅ Tải TF-IDF Baseline model thành công.")
            except Exception as exc:
                logger.warning(f"⚠️ Không thể tải TF-IDF model: {exc}")
                
        return PipelineResources(
            llm=llm,
            searcher=searcher,
            tfidf_model=tfidf_model,
            input_processor=input_processor
        )

    def run(self, source: str) -> PipelineResult:
        logger.info(f"🚀 [Pipeline] BẮT ĐẦU VẬN HÀNH: Đọc nguồn '{source[:60]}...'")

        # [Stage 1] Bộ tải Raw Text từ URI hoặc Cào Web
        stage1 = self.input_processor.process(source)
        result = PipelineResult(
            source=stage1["source"],
            raw_text=stage1["raw_text"],
            input_type=stage1["input_type"],
            metadata=stage1["metadata"],
        )

        # Lỗi cơ học, nghỉ luôn từ đầu
        if not result.raw_text.strip():
            logger.error("🛑 [Pipeline] BÁO ĐỘNG HỦY TRUNG TÂM: Bài viết không có chữ nào.")
            result.investigation_summary = "Lỗi nghiêm trọng: Không thể trích xuất văn bản."
            return result

        # Bơm qua từng cục tẩy rửa LLM (từ 2 đến 8) - Tấm khiên chắn Error mạnh nhất
        for stage in self.stages:
            try:
                result = stage.process(result)
            except Exception as exc:
                stage_name = type(stage).__name__
                logger.error(f"🛑 [Pipeline] LỖI SỤP STAGE '{stage_name}': Bật chế độ đi ngang... Lỗi: {exc}")

        verdict = result.verity_report.conclusion if result.verity_report else 'N/A'
        logger.info(f"✅ [Pipeline] XONG QUY TRÌNH. QUYẾT ĐỊNH (VERDICT) CUỐI CÙNG: {verdict}")
        
        return result
