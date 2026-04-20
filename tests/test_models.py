import pytest
from pydantic import ValidationError
from sequential_adversarial.models import (
    Claim, InvestigationResult, SourceCheck, ClaimAnalysis, 
    AnalysisResult, BiasReport, VerityReport, PipelineResult,
    TFIDFComparison
)

def test_claim_validation():
    # Valid claim
    c = Claim(text="Test", suspicion_score=0.5, loaded_language=["bad"])
    assert c.text == "Test"
    
    # Invalid score (too high)
    with pytest.raises(ValidationError):
        Claim(text="Test", suspicion_score=1.5)
    
    # Invalid score (too low)
    with pytest.raises(ValidationError):
        Claim(text="Test", suspicion_score=-0.1)

def test_claim_defaults():
    c = Claim(text="Test", suspicion_score=0.0)
    assert c.loaded_language == []

def test_investigation_result_defaults():
    ir = InvestigationResult()
    assert ir.claims == []
    assert ir.overall_manipulation_score == 0.0
    assert ir.summary == ""

def test_source_check_stance():
    # Valid stances
    for stance in ["support", "refute", "neutral"]:
        sc = SourceCheck(url="http://test.com", stance=stance, reliability=0.5)
        assert sc.stance == stance
    
    # Invalid stance
    with pytest.raises(ValidationError):
        SourceCheck(url="http://test.com", stance="maybe", reliability=0.5)

def test_claim_analysis_verdict():
    c = Claim(text="X", suspicion_score=0)
    # Valid verdicts
    for v in ["supported", "refuted", "mixed", "unverified"]:
        ca = ClaimAnalysis(claim=c, verdict=v)
        assert ca.verdict == v
    
    # Invalid verdict
    with pytest.raises(ValidationError):
        ClaimAnalysis(claim=c, verdict="unknown")

def test_verity_report_conclusion():
    # Valid conclusions
    for c in ["True", "False", "Mixed"]:
        vr = VerityReport(conclusion=c, confidence=0.8)
        assert vr.conclusion == c
    
    # Invalid conclusion
    with pytest.raises(ValidationError):
        VerityReport(conclusion="Maybe", confidence=0.5)

def test_pipeline_result_serialization():
    res = PipelineResult(source="test", raw_text="content", input_type="raw")
    json_data = res.model_dump_json()
    assert '"source":"test"' in json_data
    
    # Round trip
    res2 = PipelineResult.model_validate_json(json_data)
    assert res2.source == res.source
    assert res2.raw_text == res.raw_text

def test_tfidf_comparison_defaults():
    tc = TFIDFComparison()
    assert tc.tfidf_label == "fake"
    assert tc.tfidf_confidence == 0.5
    assert tc.agreement is False
