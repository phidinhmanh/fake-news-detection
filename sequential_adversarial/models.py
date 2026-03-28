"""
models.py — Pydantic Stage Models for Sequential Adversarial Pipeline
======================================================================
Typed data contracts for each stage's output. Ensures type safety and
makes serialization/deserialization straightforward.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


# ── Stage 1: Input ─────────────────────────────────────────────────────────────
class InputData(BaseModel):
    """Output of Stage 1 (InputProcessor)."""
    source: str = Field(..., description="Original source string (URL, file path, or raw text)")
    raw_text: str = Field(..., description="Extracted plain text content")
    input_type: Literal["url", "file", "raw"] = Field(..., description="Detected input type")
    metadata: dict = Field(default_factory=dict, description="Additional metadata (e.g. title, domain)")


# ── Stage 2: Lead Investigator ─────────────────────────────────────────────────
class Claim(BaseModel):
    """A single factual claim extracted from the text."""
    text: str = Field(..., description="The claim text")
    suspicion_score: float = Field(..., ge=0.0, le=1.0, description="How suspicious this claim is (0=benign, 1=very suspicious)")
    loaded_language: list[str] = Field(default_factory=list, description="Emotionally manipulative words/phrases found in the claim")


class InvestigationResult(BaseModel):
    """Output of Stage 2 (LeadInvestigator)."""
    claims: list[Claim] = Field(default_factory=list)
    overall_manipulation_score: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str = Field(default="", description="Brief summary of the investigation")


# ── Stage 3: Data Analyst ──────────────────────────────────────────────────────
class SourceCheck(BaseModel):
    """Result of checking a claim against one source."""
    url: str = Field(..., description="Source URL or identifier")
    stance: Literal["support", "refute", "neutral"] = Field(..., description="Whether this source supports or refutes the claim")
    reliability: float = Field(..., ge=0.0, le=1.0, description="Estimated reliability of the source")
    excerpt: str = Field(default="", description="Relevant excerpt from the source")


class ClaimAnalysis(BaseModel):
    """Cross-check result for a single claim."""
    claim: Claim
    sources: list[SourceCheck] = Field(default_factory=list)
    verdict: Literal["supported", "refuted", "mixed", "unverified"] = Field(default="unverified")


class AnalysisResult(BaseModel):
    """Output of Stage 3 (DataAnalyst)."""
    claim_analyses: list[ClaimAnalysis] = Field(default_factory=list)
    summary: str = Field(default="")


# ── Stage 4: Bias Auditor ──────────────────────────────────────────────────────
class BiasReport(BaseModel):
    """Output of Stage 4 (BiasAuditor)."""
    framing: str = Field(default="", description="Detected narrative framing (e.g. 'Economic optimism', 'Political fear')")
    distortion_detected: bool = Field(default=False)
    distortion_type: Optional[str] = Field(default=None, description="Type of distortion (e.g. 'cherry-picking', 'false urgency')")
    adversarial_notes: str = Field(default="", description="Counter-arguments and challenges to the analyst's findings")
    analyst_blind_spots: list[str] = Field(default_factory=list)


# ── Stage 5: Synthesizer ───────────────────────────────────────────────────────
class VerityReport(BaseModel):
    """Output of Stage 5 (Synthesizer) — the final verdict."""
    conclusion: Literal["True", "False", "Mixed"] = Field(..., description="Final classification")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_summary: str = Field(default="")
    bias_summary: str = Field(default="")
    key_findings: list[str] = Field(default_factory=list)
    markdown_report: str = Field(default="", description="Full Markdown-formatted report")


# ── Stage 8: TF-IDF Comparison ────────────────────────────────────────────────
class TFIDFComparison(BaseModel):
    """Output of Stage 8 (TF-IDF Comparator) — baseline comparison."""
    tfidf_label: Literal["fake", "real"] = Field(default="fake", description="TF-IDF predicted label")
    tfidf_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="TF-IDF confidence score")
    fake_proba: float = Field(default=0.5, ge=0.0, le=1.0, description="P(fake) from TF-IDF")
    real_proba: float = Field(default=0.5, ge=0.0, le=1.0, description="P(real) from TF-IDF")
    llm_verdict: Optional[Literal["True", "False", "Mixed"]] = Field(
        default=None, description="LLM pipeline's verdict"
    )
    llm_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="LLM confidence")
    agreement: bool = Field(default=False, description="Do LLM and TF-IDF agree?")
    disagreement_notes: str = Field(
        default="", description="Analysis when LLM and TF-IDF disagree"
    )


# ── Full Pipeline Result ───────────────────────────────────────────────────────
class PipelineResult(BaseModel):
    """Complete output of all 7 pipeline stages."""
    # Input
    source: str = ""
    raw_text: str = ""
    input_type: str = "raw"
    metadata: dict = Field(default_factory=dict)

    # Stage 2
    claims: list[Claim] = Field(default_factory=list)
    overall_manipulation_score: float = 0.0
    investigation_summary: str = ""

    # Stage 3
    claim_analyses: list[ClaimAnalysis] = Field(default_factory=list)
    analysis_summary: str = ""

    # Stage 4
    bias_report: Optional[BiasReport] = None

    # Stage 5
    verity_report: Optional[VerityReport] = None

    # Stage 6
    visual_flowchart_path: Optional[str] = None
    mermaid_diagram: Optional[str] = None

    # Stage 7
    saved_id: Optional[str] = None
    db_record_id: Optional[int] = None

    # Stage 8
    tfidf_comparison: Optional[TFIDFComparison] = None
