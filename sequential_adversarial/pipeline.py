"""
pipeline.py — Sequential Adversarial Pipeline (8 Stages)
=========================================================
Each stage is an encapsulated class. The orchestrator `SequentialAdversarialPipeline`
chains them in order and returns a `PipelineResult`.

Stages:
  1. InputProcessor      — text extraction (in input_processor.py)
  2. LeadInvestigator    — claim extraction via LLM
  3. DataAnalyst         — source cross-checking via LLM
  4. BiasAuditor         — adversarial bias detection via LLM
  5. Synthesizer         — final verdict generation via LLM
  6. VisualEngine        — Mermaid flowchart generation
  7. Persistence         — SQLite storage
  8. TFIDFComparator     — compare LLM verdict against TF-IDF baseline

Both real (Gemini) and mock (no-API-key) modes are supported.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from sequential_adversarial.input_processor import InputProcessor
from sequential_adversarial.llm_client import LLMClient, extract_json
from sequential_adversarial.models import (
    Claim,
    ClaimAnalysis,
    SourceCheck,
    InvestigationResult,
    AnalysisResult,
    BiasReport,
    VerityReport,
    PipelineResult,
    TFIDFComparison,
)

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────────
# Stage 2 — Lead Investigator
# ───────────────────────────────────────────────────────────────────────────────
class LeadInvestigator:
    """
    Identifies core factual claims in the article text.
    Scores each claim for suspicion and flags loaded language.
    """

    PROMPT_TEMPLATE = """You are a lead investigative journalist specialising in fact-checking.

Analyse the following article text and extract 2-4 core factual claims worth investigating.
For each claim:
- Write the claim text clearly.
- Score suspicion from 0.0 (completely benign) to 1.0 (very suspicious).
- List emotionally loaded or manipulative words/phrases you found.

Also provide an overall manipulation score (0.0-1.0) and a brief investigation summary.

Respond ONLY with valid JSON in this exact schema:
{{
  "claims": [
    {{
      "text": "string",
      "suspicion_score": 0.0,
      "loaded_language": ["word1", "word2"]
    }}
  ],
  "overall_manipulation_score": 0.0,
  "summary": "string"
}}

--- ARTICLE TEXT ---
{text}
"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def process(self, result: PipelineResult) -> PipelineResult:
        logger.info("[Stage 2] LeadInvestigator: extracting claims...")
        prompt = self.PROMPT_TEMPLATE.format(text=result.raw_text[:4000])
        raw = self.llm.generate(prompt, stage_key="lead_investigator")
        data = extract_json(raw)

        try:
            inv = InvestigationResult(**data)
        except Exception as exc:
            logger.warning(f"[Stage 2] Failed to parse LLM response: {exc}. Using fallback.")
            inv = InvestigationResult(
                claims=[Claim(text=result.raw_text[:200], suspicion_score=0.5, loaded_language=[])],
                overall_manipulation_score=0.5,
                summary="Could not parse LLM response — using fallback.",
            )

        result.claims = inv.claims
        result.overall_manipulation_score = inv.overall_manipulation_score
        result.investigation_summary = inv.summary
        return result


# ───────────────────────────────────────────────────────────────────────────────
# Stage 3 — Data Analyst
# ───────────────────────────────────────────────────────────────────────────────
class DataAnalyst:
    """
    Cross-checks each claim against plausible real-world sources.
    Returns a verdict (supported/refuted/mixed/unverified) per claim.
    """

    PROMPT_TEMPLATE = """You are a senior data analyst at a fact-checking organisation.

Your job is to cross-check the following claims against known credible sources.
For EACH claim provide 2-3 hypothetical source references (realistic news agencies, 
official reports, academic papers) and state whether each source supports, refutes, 
or is neutral about the claim. Estimate source reliability (0.0-1.0).

Respond ONLY with valid JSON:
{{
  "claim_analyses": [
    {{
      "claim": {{
        "text": "string",
        "suspicion_score": 0.0,
        "loaded_language": []
      }},
      "sources": [
        {{
          "url": "string",
          "stance": "support|refute|neutral",
          "reliability": 0.0,
          "excerpt": "string"
        }}
      ],
      "verdict": "supported|refuted|mixed|unverified"
    }}
  ],
  "summary": "string"
}}

--- CLAIMS TO ANALYSE ---
{claims_json}
"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def process(self, result: PipelineResult) -> PipelineResult:
        logger.info("[Stage 3] DataAnalyst: cross-checking claims...")
        claims_json = json.dumps(
            [c.model_dump() for c in result.claims], ensure_ascii=False, indent=2
        )
        prompt = self.PROMPT_TEMPLATE.format(claims_json=claims_json)
        raw = self.llm.generate(prompt, stage_key="data_analyst")
        data = extract_json(raw)

        try:
            analysis = AnalysisResult(**data)
        except Exception as exc:
            logger.warning(f"[Stage 3] Failed to parse: {exc}. Using fallback.")
            analysis = AnalysisResult(
                claim_analyses=[
                    ClaimAnalysis(claim=claim, sources=[], verdict="unverified")
                    for claim in result.claims
                ],
                summary="Source analysis failed — could not parse LLM response.",
            )

        result.claim_analyses = analysis.claim_analyses
        result.analysis_summary = analysis.summary
        return result


# ───────────────────────────────────────────────────────────────────────────────
# Stage 4 — Bias Auditor
# ───────────────────────────────────────────────────────────────────────────────
class BiasAuditor:
    """
    Adversarial agent that challenges the analyst's findings.
    Identifies framing bias, distortion, and blind spots.
    """

    PROMPT_TEMPLATE = """You are an adversarial bias auditor whose job is to challenge 
the previous analyst's findings and identify framing bias, narrative distortion, 
and unexamined assumptions.

Given:
- Article text (excerpt)
- Claims identified
- Analyst's source analysis

Your task:
1. Identify the overall narrative framing (e.g. "Economic fear", "Political conspiracy").
2. Detect if any distortion is present (cherry-picking, false urgency, appeal to authority, etc.).
3. Write adversarial notes challenging the analyst's conclusions.
4. List the analyst's blind spots.

Respond ONLY with valid JSON:
{{
  "framing": "string",
  "distortion_detected": true,
  "distortion_type": "string or null",
  "adversarial_notes": "string",
  "analyst_blind_spots": ["string"]
}}

--- ARTICLE EXCERPT ---
{text}

--- ANALYST FINDINGS ---
{analysis_json}
"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def process(self, result: PipelineResult) -> PipelineResult:
        logger.info("[Stage 4] BiasAuditor: auditing for bias and framing...")
        analysis_json = json.dumps(
            [ca.model_dump() for ca in result.claim_analyses],
            ensure_ascii=False,
            indent=2,
        )
        prompt = self.PROMPT_TEMPLATE.format(
            text=result.raw_text[:2000],
            analysis_json=analysis_json,
        )
        raw = self.llm.generate(prompt, stage_key="bias_auditor")
        data = extract_json(raw)

        try:
            bias = BiasReport(**data)
        except Exception as exc:
            logger.warning(f"[Stage 4] Failed to parse: {exc}. Using fallback.")
            bias = BiasReport(
                framing="Unknown",
                distortion_detected=False,
                adversarial_notes="Bias analysis failed.",
                analyst_blind_spots=[],
            )

        result.bias_report = bias
        return result


# ───────────────────────────────────────────────────────────────────────────────
# Stage 5 — Synthesizer
# ───────────────────────────────────────────────────────────────────────────────
class Synthesizer:
    """
    Produces the final Verity Report by synthesising all upstream findings.
    """

    PROMPT_TEMPLATE = """You are the Chief Verification Officer. Based on all the 
evidence collected, produce a final Verity Report.

Evidence:
- Claims and suspicion scores
- Source verdicts from the analyst
- Bias audit from the adversarial auditor

Your report must:
1. State a clear conclusion: "True", "False", or "Mixed"
2. Give a confidence score (0.0-1.0)
3. Summarise the evidence
4. Summarise the bias/framing
5. List 3-5 key findings
6. Write a full Markdown-formatted report

Respond ONLY with valid JSON:
{{
  "conclusion": "True|False|Mixed",
  "confidence": 0.0,
  "evidence_summary": "string",
  "bias_summary": "string",
  "key_findings": ["string"],
  "markdown_report": "string"
}}

--- INVESTIGATION DATA ---
Claims: {claims_json}
Analysis: {analysis_json}
Bias Report: {bias_json}
"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def process(self, result: PipelineResult) -> PipelineResult:
        logger.info("[Stage 5] Synthesizer: generating Verity Report...")
        prompt = self.PROMPT_TEMPLATE.format(
            claims_json=json.dumps(
                [c.model_dump() for c in result.claims], ensure_ascii=False
            ),
            analysis_json=json.dumps(
                [ca.model_dump() for ca in result.claim_analyses], ensure_ascii=False
            ),
            bias_json=(
                result.bias_report.model_dump_json()
                if result.bias_report
                else "{}"
            ),
        )
        raw = self.llm.generate(prompt, stage_key="synthesizer")
        data = extract_json(raw)

        try:
            verity = VerityReport(**data)
        except Exception as exc:
            logger.warning(f"[Stage 5] Failed to parse: {exc}. Using fallback.")
            verity = VerityReport(
                conclusion="Mixed",
                confidence=0.5,
                evidence_summary="Synthesis failed — could not parse LLM response.",
                markdown_report="## Verity Report\n**Status:** Error during synthesis.",
            )

        result.verity_report = verity
        return result


# ───────────────────────────────────────────────────────────────────────────────
# Stage 6 — Visual Engine
# ───────────────────────────────────────────────────────────────────────────────
class VisualEngine:
    """
    Generates a Mermaid.js flowchart from the pipeline results.
    Saves a .md file with an embedded Mermaid code block.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        from config import SA_VISUAL_DIR
        self.output_dir = output_dir or SA_VISUAL_DIR

    def process(self, result: PipelineResult) -> PipelineResult:
        logger.info("[Stage 6] VisualEngine: generating Mermaid flowchart...")
        diagram = self._build_diagram(result)
        result.mermaid_diagram = diagram

        # Save to file
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = self.output_dir / f"verity_flowchart_{ts}.md"
            md_content = f"# Verity Report — Evidence Flowchart\n\n```mermaid\n{diagram}\n```\n"
            out_path.write_text(md_content, encoding="utf-8")
            result.visual_flowchart_path = str(out_path)
            logger.info(f"[Stage 6] Flowchart saved to {out_path}")
        except Exception as exc:
            logger.warning(f"[Stage 6] Could not save flowchart: {exc}")

        return result

    def _build_diagram(self, result: PipelineResult) -> str:
        """Build a Mermaid flowchart from pipeline data."""
        lines = ["flowchart TD"]
        lines.append(f'    IN["📰 Article Input\\n{self._truncate(result.source, 40)}"]')
        lines.append('    IN --> INV["🔍 Lead Investigator"]')

        # Claims
        for i, claim in enumerate(result.claims[:3]):
            cid = f"C{i}"
            label = self._truncate(claim.text, 35)
            score = f"{claim.suspicion_score:.0%}"
            lines.append(f'    INV --> {cid}["⚠️ Claim {i+1}: {label}\\nSuspicion: {score}"]')

            # Verdicts from claim analyses
            if i < len(result.claim_analyses):
                verdict = result.claim_analyses[i].verdict.upper()
                emoji = {"REFUTED": "❌", "SUPPORTED": "✅", "MIXED": "🟡", "UNVERIFIED": "❓"}.get(verdict, "❓")
                lines.append(f'    {cid} --> V{i}["{emoji} {verdict}"]')

        # Bias
        if result.bias_report:
            distortion = "⚠️ Bias Detected" if result.bias_report.distortion_detected else "✅ No Bias"
            framing = self._truncate(result.bias_report.framing, 30)
            lines.append(f'    INV --> BIAS["🧐 Bias Auditor\\n{distortion}\\nFraming: {framing}"]')

        # Final verdict
        if result.verity_report:
            conclusion = result.verity_report.conclusion
            emoji = {"True": "✅", "False": "❌", "Mixed": "🟡"}.get(conclusion, "❓")
            confidence = f"{result.verity_report.confidence:.0%}"
            lines.append(
                f'    BIAS --> FINAL["{emoji} VERDICT: {conclusion}\\nConfidence: {confidence}"]'
            )
            lines.append(f'    FINAL --> DB["💾 Saved to Database"]')

        return "\n".join(lines)

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        text = str(text).replace('"', "'").replace("\n", " ")
        return text[:max_len] + "..." if len(text) > max_len else text


# ───────────────────────────────────────────────────────────────────────────────
# Stage 7 — Persistence
# ───────────────────────────────────────────────────────────────────────────────
class Persistence:
    """
    Saves the PipelineResult to a SQLite database for long-term memory.
    """

    def __init__(self, db_path: Optional[Path] = None):
        from config import SA_DB_PATH
        self.db_path = db_path or SA_DB_PATH
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Create the database and table if they don't exist."""
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
        logger.info("[Stage 7] Persistence: saving to database...")
        conclusion = result.verity_report.conclusion if result.verity_report else "Unknown"
        confidence = result.verity_report.confidence if result.verity_report else 0.0
        created_at = datetime.now().isoformat()
        full_json = result.model_dump_json(indent=None)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """INSERT INTO reports
                       (source, conclusion, confidence, manipulation_score, full_json, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        result.source,
                        conclusion,
                        confidence,
                        result.overall_manipulation_score,
                        full_json,
                        created_at,
                    ),
                )
                record_id = cursor.lastrowid
                conn.commit()

            result.db_record_id = record_id
            result.saved_id = f"db_{record_id}"
            logger.info(f"[Stage 7] Saved as record #{record_id} in {self.db_path}")
        except Exception as exc:
            logger.error(f"[Stage 7] Database save failed: {exc}")

        return result

    def fetch_all(self) -> list[dict]:
        """Retrieve all saved reports (lightweight view, no full_json)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, source, conclusion, confidence, manipulation_score, created_at "
                "FROM reports ORDER BY id DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def fetch_by_id(self, record_id: int) -> Optional[dict]:
        """Retrieve a full report by DB id."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM reports WHERE id = ?", (record_id,)
            ).fetchone()
        return dict(row) if row else None


# ───────────────────────────────────────────────────────────────────────────────
# Stage 8 — TF-IDF Comparator
# ───────────────────────────────────────────────────────────────────────────────
class TFIDFComparator:
    """
    Compares the LLM pipeline's verdict against the TF-IDF baseline.
    Runs after Stage 5 (Synthesizer) and before Stage 7 (Persistence).
    """

    def __init__(self, mock: bool = False):
        self.mock = mock
        self._model = None

    def _load_model(self):
        """Lazily load the baseline TF-IDF model."""
        if self._model is None:
            try:
                from model.baseline_logreg import BaselineLogReg

                model = BaselineLogReg()
                model.load()
                self._model = model
            except Exception as exc:
                logger.warning(f"[Stage 8] Could not load TF-IDF model: {exc}")
                self._model = None

    def process(self, result: PipelineResult) -> PipelineResult:
        logger.info("[Stage 8] TF-IDF Comparator: comparing LLM vs baseline...")

        if self.mock:
            # Return mock comparison
            tfidf_label = "fake"
            fake_proba = 0.72
            real_proba = 0.28
        else:
            self._load_model()
            if self._model is None:
                logger.warning("[Stage 8] TF-IDF model unavailable. Skipping comparison.")
                return result

            try:
                scores = self._model.predict_with_score(result.raw_text)
                tfidf_label = scores["label"]
                fake_proba = scores["fake_proba"]
                real_proba = scores["real_proba"]
            except Exception as exc:
                logger.warning(f"[Stage 8] TF-IDF prediction failed: {exc}")
                return result

        llm_verdict = result.verity_report.conclusion if result.verity_report else None
        llm_confidence = result.verity_report.confidence if result.verity_report else 0.0

        # Determine agreement: LLM False/Mixed ≈ TF-IDF fake, LLM True ≈ TF-IDF real
        llm_predicted_fake = llm_verdict in ("False", "Mixed")
        tfidf_predicted_fake = tfidf_label == "fake"
        agreement = llm_predicted_fake == tfidf_predicted_fake

        if not agreement:
            notes = (
                f"LLM classified as '{llm_verdict}' but TF-IDF predicted '{tfidf_label}' "
                f"(TF-IDF fake_prob={fake_proba:.2f}). "
                "This disagreement may warrant human review."
            )
        else:
            notes = "LLM and TF-IDF agree on the classification."

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
# Orchestrator
# ───────────────────────────────────────────────────────────────────────────────
class SequentialAdversarialPipeline:
    """
    7-stage Sequential Adversarial Pipeline orchestrator.

    Usage:
        pipeline = SequentialAdversarialPipeline()
        result = pipeline.run("https://example.com/some-news-article")
        print(result.verity_report.conclusion)   # "True" / "False" / "Mixed"
        print(result.verity_report.markdown_report)
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        mock: bool = False,
    ):
        self.llm = llm or LLMClient(mock=mock)
        self.input_processor = InputProcessor()
        self.stages = [
            LeadInvestigator(self.llm),
            DataAnalyst(self.llm),
            BiasAuditor(self.llm),
            Synthesizer(self.llm),
            VisualEngine(),
            Persistence(),
            TFIDFComparator(mock=mock),
        ]

    def run(self, source: str) -> PipelineResult:
        """
        Execute all 7 stages for the given source.

        Args:
            source: URL, file path, or raw article text.

        Returns:
            PipelineResult with all stage outputs populated.
        """
        logger.info(f"[Pipeline] Starting for source: {source[:80]}...")

        # Stage 1 — Input Processing
        stage1 = self.input_processor.process(source)
        result = PipelineResult(
            source=stage1["source"],
            raw_text=stage1["raw_text"],
            input_type=stage1["input_type"],
            metadata=stage1["metadata"],
        )

        if not result.raw_text.strip():
            logger.error("[Pipeline] No text extracted from source. Aborting.")
            result.investigation_summary = "Error: Could not extract text from source."
            return result

        # Stages 2-7 — transform result in place
        for stage in self.stages:
            try:
                result = stage.process(result)
            except Exception as exc:
                stage_name = type(stage).__name__
                logger.error(f"[Pipeline] {stage_name} failed: {exc}", exc_info=True)

        logger.info(
            f"[Pipeline] Done. Verdict: {result.verity_report.conclusion if result.verity_report else 'N/A'}"
        )
        return result
