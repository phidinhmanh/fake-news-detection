import pytest
import json
import pandas as pd
from unittest.mock import MagicMock, patch
from agents.knowledge_base import KnowledgeBase
from agents.evidence_searcher import EvidenceSearcher, EvidenceItem

@pytest.fixture
def mock_kb_resources():
    with patch("sentence_transformers.SentenceTransformer") as mock_st, \
         patch("lancedb.connect") as mock_ldb:
        
        # Mock encoder
        encoder = mock_st.return_value
        encoder.encode.return_value = MagicMock()
        
        # Mock table search
        db = mock_ldb.return_value
        table = MagicMock()
        db.open_table.return_value = table
        db.table_names.return_value = ["vifactcheck_evidence"]
        
        # Mock search result (pandas df)
        mock_df = pd.DataFrame([{
            "claim": "claim 1",
            "evidence": "evidence 1",
            "label": "supported",
            "_distance": 0.1
        }])
        table.search.return_value.limit.return_value.to_pandas.return_value = mock_df
        
        yield db, table, encoder

def test_kb_initialization(tmp_path):
    kb = KnowledgeBase(db_path=tmp_path)
    assert kb.db_path == tmp_path
    assert tmp_path.exists()

def test_kb_search_mocked(tmp_path, mock_kb_resources):
    kb = KnowledgeBase(db_path=tmp_path)
    results = kb.search("test query")
    assert len(results) == 1
    assert results[0]["claim"] == "claim 1"
    assert results[0]["score"] == pytest.approx(0.1)

@patch("requests.get")
def test_searcher_wiki(mock_get):
    # Mock wiki search response
    mock_get.return_value.json.side_effect = [
        {"query": {"search": [{"title": "WikiTitle"}]}}, # Search
        {"query": {"pages": {"123": {"extract": "Wiki Extract"}}}}, # Extract
    ]
    
    searcher = EvidenceSearcher(use_wiki=True, use_serper=False)
    results = searcher._wiki_search("query")
    assert len(results) == 1
    assert "WikiTitle" in results[0]["text"]

def test_search_claim_sorting():
    # Mock KB return
    mock_kb = MagicMock()
    mock_kb.search.return_value = [
        {"claim": "c", "evidence": "e1", "label": "L", "score": 0.2},
        {"claim": "c", "evidence": "e2", "label": "L", "score": 0.4}
    ]
    
    searcher = EvidenceSearcher(kb=mock_kb, use_wiki=False)
    results = searcher.search_claim("query")
    
    assert len(results) == 2
    # Sorted by score (1-distance) descending
    # e1 score: 1 - 0.2 = 0.8
    # e2 score: 1 - 0.4 = 0.6
    assert results[0].text == "e1"

def test_searcher_multi():
    mock_kb = MagicMock()
    mock_kb.search.return_value = [{"claim": "c", "evidence": "e", "label": "L", "score": 0.1}]
    searcher = EvidenceSearcher(kb=mock_kb, use_wiki=False)
    
    results = searcher.search_multi(["claim 1", "claim 2"])
    assert len(results) == 2
    assert "claim 1" in results
    assert "claim 2" in results
    assert len(results["claim 1"]) == 1

def test_searcher_single_claim():
    mock_kb = MagicMock()
    mock_kb.search.return_value = []
    searcher = EvidenceSearcher(kb=mock_kb, use_wiki=False)
    # This just ensures no crash and valid return
    res = searcher.search_claim("test")
    assert isinstance(res, list)

def test_kb_initialization_no_db(tmp_path):
    # Test initialization when DB doesn't exist yet
    db_path = tmp_path / "new_kb_dir" # Better not use .db suffix for a directory
    from agents.knowledge_base import KnowledgeBase
    kb = KnowledgeBase(db_path=db_path)
    assert kb.db_path == db_path
    assert db_path.exists() # Should be created by __init__

def test_kb_build_from_vifactcheck_mock(tmp_path):
    # Mocking jsonl file and lancedb
    jsonl = tmp_path / "test.jsonl"
    with open(jsonl, "w") as f:
        f.write('{"claim": "test", "evidence": "evidence"}\n')
    
    from agents.knowledge_base import KnowledgeBase
    kb = KnowledgeBase(db_path=tmp_path)
    with patch.object(kb, "_get_resources") as mock_res:
        mock_db = MagicMock()
        mock_table = MagicMock()
        mock_res.return_value = (mock_db, mock_table, MagicMock())
        
        count = kb.build_from_vifactcheck(jsonl)
        assert count == 1
