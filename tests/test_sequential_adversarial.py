"""
test_sequential_adversarial.py
================================
Tests for the 7-stage Sequential Adversarial Pipeline.
All tests run in mock mode (no API key required).
"""

import logging
import sqlite3
from pathlib import Path

import pytest

from sequential_adversarial.input_processor import InputProcessor
from sequential_adversarial.llm_client import LLMClient, extract_json
from sequential_adversarial.models import (
    Claim,
    InvestigationResult,
    VerityReport,
    PipelineResult,
)
from sequential_adversarial.pipeline import (
    SequentialAdversarialPipeline,
    LeadInvestigator,
    DataAnalyst,
    BiasAuditor,
    Synthesizer,
    VisualEngine,
    Persistence,
)

logging.basicConfig(level=logging.INFO)


# ── Fixtures ────────────────────────────────────────────────────────────────────
@pytest.fixture()
def mock_llm() -> LLMClient:
    return LLMClient(mock=True)


@pytest.fixture()
def tmp_db(tmp_path) -> Path:
    return tmp_path / "test_reports.db"


@pytest.fixture()
def base_result() -> PipelineResult:
    return PipelineResult(
        source="https://test.com/article",
        raw_text=(
            "The stock market will crash by 50% next week! "
            "Government officials are hiding the truth from the public."
        ),
        input_type="url",
        metadata={"domain": "test.com"},
    )


@pytest.fixture()
def full_mock_pipeline(tmp_path, tmp_db) -> SequentialAdversarialPipeline:
    """Pipeline with mock LLM and temp directories."""
    pipeline = SequentialAdversarialPipeline(mock=True)
    # Override Persistence and VisualEngine to use tmp dirs
    vis = VisualEngine(output_dir=tmp_path / "visuals")
    pers = Persistence(db_path=tmp_db)
    pipeline.stages[-2] = vis
    pipeline.stages[-1] = pers
    return pipeline


# ── Stage 1: InputProcessor ────────────────────────────────────────────────────
class TestInputProcessor:
    def test_raw_text(self):
        ip = InputProcessor()
        out = ip.process("Breaking news: scientists discover new species")
        assert out["input_type"] == "raw"
        assert "Breaking news" in out["raw_text"]
        assert out["source"] == "Breaking news: scientists discover new species"
        assert "metadata" in out

    def test_file_txt(self, tmp_path):
        test_file = tmp_path / "article.txt"
        test_file.write_text("This is a test article about fake news.", encoding="utf-8")
        ip = InputProcessor()
        out = ip.process(str(test_file))
        assert out["input_type"] == "file"
        assert "test article" in out["raw_text"]
        assert out["metadata"]["filename"] == "article.txt"

    def test_url_detection(self):
        ip = InputProcessor()
        out = ip.process("https://example.com/article")
        assert out["input_type"] == "url"
        # URL may fail to fetch in test env — ensure we get a result dict anyway
        assert "raw_text" in out

    def test_max_chars_limit(self):
        ip = InputProcessor()
        long_text = "A" * 10_000
        out = ip.process(long_text)
        assert len(out["raw_text"]) <= InputProcessor.MAX_CHARS


# ── Stage 2: LeadInvestigator ──────────────────────────────────────────────────
class TestLeadInvestigator:
    def test_extracts_claims(self, mock_llm, base_result):
        stage = LeadInvestigator(mock_llm)
        result = stage.process(base_result)
        assert len(result.claims) > 0
        claim = result.claims[0]
        assert 0.0 <= claim.suspicion_score <= 1.0
        assert isinstance(claim.loaded_language, list)

    def test_manipulation_score_range(self, mock_llm, base_result):
        stage = LeadInvestigator(mock_llm)
        result = stage.process(base_result)
        assert 0.0 <= result.overall_manipulation_score <= 1.0

    def test_summary_populated(self, mock_llm, base_result):
        stage = LeadInvestigator(mock_llm)
        result = stage.process(base_result)
        assert result.investigation_summary.strip() != ""


# ── Stage 3: DataAnalyst ───────────────────────────────────────────────────────
class TestDataAnalyst:
    def test_produces_verdicts(self, mock_llm, base_result):
        # First run Stage 2 so claims are populated
        base_result = LeadInvestigator(mock_llm).process(base_result)
        stage = DataAnalyst(mock_llm)
        result = stage.process(base_result)
        assert len(result.claim_analyses) > 0
        for ca in result.claim_analyses:
            assert ca.verdict in ("supported", "refuted", "mixed", "unverified")

    def test_sources_have_reliability(self, mock_llm, base_result):
        base_result = LeadInvestigator(mock_llm).process(base_result)
        result = DataAnalyst(mock_llm).process(base_result)
        for ca in result.claim_analyses:
            for src in ca.sources:
                assert 0.0 <= src.reliability <= 1.0


# ── Stage 4: BiasAuditor ───────────────────────────────────────────────────────
class TestBiasAuditor:
    def test_bias_report_populated(self, mock_llm, base_result):
        base_result = LeadInvestigator(mock_llm).process(base_result)
        base_result = DataAnalyst(mock_llm).process(base_result)
        result = BiasAuditor(mock_llm).process(base_result)
        assert result.bias_report is not None
        assert result.bias_report.framing.strip() != ""
        assert isinstance(result.bias_report.adversarial_notes, str)
        assert isinstance(result.bias_report.analyst_blind_spots, list)


# ── Stage 5: Synthesizer ───────────────────────────────────────────────────────
class TestSynthesizer:
    def test_verdict_is_valid(self, mock_llm, base_result):
        base_result = LeadInvestigator(mock_llm).process(base_result)
        base_result = DataAnalyst(mock_llm).process(base_result)
        base_result = BiasAuditor(mock_llm).process(base_result)
        result = Synthesizer(mock_llm).process(base_result)
        assert result.verity_report is not None
        assert result.verity_report.conclusion in ("True", "False", "Mixed")
        assert 0.0 <= result.verity_report.confidence <= 1.0

    def test_markdown_report_generated(self, mock_llm, base_result):
        base_result = LeadInvestigator(mock_llm).process(base_result)
        base_result = DataAnalyst(mock_llm).process(base_result)
        base_result = BiasAuditor(mock_llm).process(base_result)
        result = Synthesizer(mock_llm).process(base_result)
        assert result.verity_report.markdown_report.strip() != ""


# ── Stage 6: VisualEngine ──────────────────────────────────────────────────────
class TestVisualEngine:
    def test_mermaid_diagram_generated(self, tmp_path, base_result):
        base_result.claims = [
            Claim(text="Claim A", suspicion_score=0.8, loaded_language=["crash"])
        ]
        base_result.verity_report = VerityReport(
            conclusion="False", confidence=0.9,
            markdown_report="## Report"
        )
        engine = VisualEngine(output_dir=tmp_path / "visuals")
        result = engine.process(base_result)
        assert result.mermaid_diagram is not None
        assert "flowchart TD" in result.mermaid_diagram

    def test_file_saved(self, tmp_path, base_result):
        base_result.verity_report = VerityReport(
            conclusion="Mixed", confidence=0.5, markdown_report=""
        )
        engine = VisualEngine(output_dir=tmp_path / "visuals")
        result = engine.process(base_result)
        if result.visual_flowchart_path:
            assert Path(result.visual_flowchart_path).exists()


# ── Stage 7: Persistence ───────────────────────────────────────────────────────
class TestPersistence:
    def test_saves_and_retrieves(self, tmp_db, base_result):
        base_result.verity_report = VerityReport(
            conclusion="True", confidence=0.95, markdown_report=""
        )
        pers = Persistence(db_path=tmp_db)
        result = pers.process(base_result)
        assert result.db_record_id is not None
        assert result.saved_id.startswith("db_")

        record = pers.fetch_by_id(result.db_record_id)
        assert record is not None
        assert record["conclusion"] == "True"
        assert record["confidence"] == pytest.approx(0.95)

    def test_fetch_all(self, tmp_db, base_result):
        base_result.verity_report = VerityReport(
            conclusion="False", confidence=0.8, markdown_report=""
        )
        pers = Persistence(db_path=tmp_db)
        pers.process(base_result)
        all_records = pers.fetch_all()
        assert len(all_records) >= 1


# ── Full Pipeline Integration ──────────────────────────────────────────────────
class TestFullPipeline:
    def test_end_to_end_raw_text(self, full_mock_pipeline):
        result = full_mock_pipeline.run(
            "Breaking: Scientists claim aliens visited Earth last Tuesday!"
        )
        # All 5 key fields should be populated
        assert result.raw_text.strip() != ""
        assert len(result.claims) > 0
        assert len(result.claim_analyses) > 0
        assert result.bias_report is not None
        assert result.verity_report is not None
        assert result.verity_report.conclusion in ("True", "False", "Mixed")
        assert result.mermaid_diagram is not None
        assert result.saved_id is not None

    def test_empty_text_handled_gracefully(self, full_mock_pipeline):
        """Pipeline should not crash on empty input."""
        result = full_mock_pipeline.run("")
        # Should return without crashing; claims may be empty
        assert isinstance(result, PipelineResult)

    def test_pipelineresult_is_serializable(self, full_mock_pipeline):
        """PipelineResult must be JSON-serializable (for API responses)."""
        import json
        result = full_mock_pipeline.run("Test article text.")
        serialized = result.model_dump_json()
        data = json.loads(serialized)
        assert isinstance(data, dict)


# ── LLMClient / extract_json ───────────────────────────────────────────────────
class TestExtractJson:
    def test_plain_json(self):
        raw = '{"foo": "bar", "num": 42}'
        result = extract_json(raw)
        assert result == {"foo": "bar", "num": 42}

    def test_markdown_code_fence(self):
        raw = '```json\n{"foo": "bar"}\n```'
        result = extract_json(raw)
        assert result == {"foo": "bar"}

    def test_fallback_to_first_object(self):
        raw = 'Here is the result: {"verdict": "True"} - hope that helps!'
        result = extract_json(raw)
        assert result.get("verdict") == "True"

    def test_malformed_returns_error_dict(self):
        with pytest.raises(ValueError):
            extract_json("this is not json at all @@@@")
