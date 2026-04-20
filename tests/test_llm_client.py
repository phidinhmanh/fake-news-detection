import pytest
import json
from sequential_adversarial.llm_client import (
    extract_json, LLMClient, OpenAICompatibleClient, 
    OllamaLLMClient, LLMClientFactory, BaseLLMProvider
)
from sequential_adversarial.models import InvestigationResult

def test_extract_json_utilities():
    # Plain
    assert extract_json('{"a": 1}') == {"a": 1}
    # Markdown
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    # Regex fallback
    assert extract_json('Text before {"a": 1} text after') == {"a": 1}
    # Invalid
    with pytest.raises(ValueError):
        extract_json("not json")

def test_gemini_client_mock():
    client = LLMClient(mock=True)
    assert client.is_mock is True
    
    resp = client.generate("hello", stage_key="lead_investigator")
    assert "claims" in resp
    
    struct = client.generate_structured("hello", InvestigationResult, stage_key="lead_investigator")
    assert isinstance(struct, InvestigationResult)
    assert len(struct.claims) > 0

def test_openai_compatible_client_mock():
    client = OpenAICompatibleClient(model_name="test", api_key="", mock=True)
    assert client.is_mock is True
    assert "claims" in client.generate("...", stage_key="lead_investigator")

def test_ollama_client_mock():
    client = OllamaLLMClient(mock=True)
    assert client.is_mock is True
    assert "claims" in client.generate("...", stage_key="lead_investigator")

def test_llm_factory():
    # Gemini
    c1 = LLMClientFactory.create("gemini", mock=True)
    assert isinstance(c1, LLMClient)
    
    # Qwen (OpenAI compatible)
    c2 = LLMClientFactory.create("qwen", mock=True)
    assert isinstance(c2, OpenAICompatibleClient)
    
    # Unknown
    with pytest.raises(ValueError):
        LLMClientFactory.create("unknown_provider")

def test_structured_fallback_to_mock(mock_llm):
    # BaseLLMProvider.generate_structured uses self.generate
    # If we force an error in a real provider but it's mock, it should still work
    # Here we just verify that generate_structured uses the mock response correctly
    struct = mock_llm.generate_structured("...", InvestigationResult, stage_key="lead_investigator")
    assert struct.overall_manipulation_score > 0
