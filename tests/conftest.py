import pytest
import pandas as pd
from pathlib import Path
from sequential_adversarial.models import (
    PipelineResult, Claim, InvestigationResult, AnalysisResult, 
    ClaimAnalysis, SourceCheck, BiasReport, VerityReport
)
from sequential_adversarial.llm_client import LLMClient

from sequential_adversarial.pipeline import (
    SequentialAdversarialPipeline,
    VisualEngine,
    Persistence,
)

@pytest.fixture()
def mock_llm() -> LLMClient:
    """Provides a mock LLM for testing."""
    return LLMClient(mock=True)

@pytest.fixture()
def full_mock_pipeline(tmp_path, tmp_db) -> SequentialAdversarialPipeline:
    """Pipeline with mock LLM and temp directories."""
    pipeline = SequentialAdversarialPipeline(mock=True)
    # Override Persistence and VisualEngine to use tmp dirs
    vis = VisualEngine(output_dir=tmp_path / "visuals")
    pers = Persistence(db_path=tmp_db)
    # Stage indices: Investigator[0], DataAnalyst[1], Bias[2], Synth[3], Visual[4], Pers[5], TFIDF[6]
    pipeline.stages[4] = vis
    pipeline.stages[5] = pers
    return pipeline

@pytest.fixture()
def base_result() -> PipelineResult:
    """Provides a basic PipelineResult with sample text."""
    return PipelineResult(
        source="https://test.com/vn-article",
        raw_text=(
            "Theo nguồn tin từ Bộ Y tế, vaccine COVID-19 hoàn toàn an toàn. "
            "Các tin đồn về tác dụng phụ nghiêm trọng là không chính xác."
        ),
        input_type="url",
        metadata={"domain": "test.com"},
    )

@pytest.fixture()
def sample_vn_df() -> pd.DataFrame:
    """Provides a sample Vietnamese dataframe for model testing."""
    data = {
        "text": [
            "Vaccine là an toàn cho mọi người.",
            "Giá xăng sẽ tăng mạnh vào ngày mai.",
            "Uống nước chanh có thể chữa khỏi ung thư.",
            "Việt Nam đã đạt được những thành tựu rực rỡ."
        ],
        "label": ["real", "real", "fake", "real"],
        "label_binary": [0, 0, 1, 0]
    }
    return pd.DataFrame(data)

@pytest.fixture()
def tmp_db(tmp_path) -> Path:
    """Provides a temporary path for a SQLite database."""
    return tmp_path / "test_reports.db"

@pytest.fixture()
def sample_claims() -> list[Claim]:
    """Provides a list of sample Claim objects."""
    return [
        Claim(text="Claim 1", suspicion_score=0.2, loaded_language=[]),
        Claim(text="Claim 2", suspicion_score=0.8, loaded_language=["nghiêm trọng"])
    ]

@pytest.fixture()
def sample_analysis(sample_claims) -> AnalysisResult:
    """Provides a sample AnalysisResult object."""
    return AnalysisResult(
        claim_analyses=[
            ClaimAnalysis(
                claim=sample_claims[0],
                sources=[SourceCheck(url="src1", stance="support", reliability=0.9, excerpt="...")],
                verdict="supported"
            ),
            ClaimAnalysis(
                claim=sample_claims[1],
                sources=[SourceCheck(url="src2", stance="refute", reliability=0.8, excerpt="...")],
                verdict="refuted"
            )
        ],
        summary="Sample analysis summary."
    )
