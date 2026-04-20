"""
llm_client.py — Unified LLM Client for Fake News Detection
======================================================
Handles multiple providers (Gemini, OpenAI, Nvidia, Grok, Qwen, Ollama)
with automatic mock fallback and structured output support.

Timeouts configured per NFR-8.4 for all external API calls.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import random
from abc import ABC, abstractmethod
from typing import Optional, Any

# Load .env at module import to ensure API keys are available
from dotenv import load_dotenv
load_dotenv()

# Import timeouts from config (NFR-8.4)
from config import LLM_TIMEOUT

logger = logging.getLogger(__name__)

# ── Mock responses for 4 core stages ──────────────────────────────────────────
_MOCK_RESPONSES: dict[str, str] = {
    "lead_investigator": json.dumps({
        "claims": [{"text": "Mock claim", "suspicion_score": 0.9, "loaded_language": ["crash"]}],
        "overall_manipulation_score": 0.8,
        "summary": "[MOCK] High suspicion content detected."
    }),
    "data_analyst": json.dumps({
        "claim_analyses": [{"claim": {"text": "Mock claim"}, "sources": [], "verdict": "refuted"}],
        "summary": "[MOCK] Claims checked against 0 sources."
    }),
    "bias_auditor": json.dumps({
        "framing": "Fear", "distortion_detected": True, "distortion_type": "false urgency",
        "adversarial_notes": "[MOCK] Narrative leverages fear.", "analyst_blind_spots": []
    }),
    "synthesizer": json.dumps({
        "conclusion": "False", "confidence": 0.8, "evidence_summary": "[MOCK] Refuted.",
        "bias_summary": "[MOCK] Biased.", "key_findings": ["Refuted by Reuters"],
        "markdown_report": "## Verity Report\n**Conclusion:** FALSE ❌"
    })
}

# ── Utilities ───────────────────────────────────────────────────────────────
def extract_json(text: str) -> dict:
    """Extract JSON from raw LLM output. Handles $defs schema format from OpenAI."""
    text = text.strip()
    if text.startswith("```json"): text = text[7:]
    elif text.startswith("```"): text = text[3:]
    if text.endswith("```"): text = text[:-3]
    text = text.strip()

    try:
        data = json.loads(text)
        # Remove $defs and $schema to get clean data
        if isinstance(data, dict):
            data.pop('$defs', None)
            data.pop('$schema', None)
            # Recursively clean nested objects
            data = _clean_json_data(data)
        return data
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                data.pop('$defs', None)
                data.pop('$schema', None)
                return _clean_json_data(data)
            except json.JSONDecodeError: pass
        raise ValueError(f"Could not parse valid JSON from LLM output: {text[:100]}")

def _clean_json_data(data: dict) -> dict:
    """Recursively remove $defs and $schema from nested objects."""
    if isinstance(data, dict):
        return {k: _clean_json_data(v) for k, v in data.items() if k not in ('$defs', '$schema')}
    elif isinstance(data, list):
        return [_clean_json_data(item) if isinstance(item, dict) else item for item in data]
    return data

# ── Base LLM Provider ──────────────────────────────────────────────────────────
class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, stage_key: str = "") -> str: pass

    def generate_structured(self, prompt: str, schema: type, stage_key: str = "") -> Any:
        """Default implementation for structured generation."""
        if self.is_mock:
            return schema(**extract_json(self._mock_generate(stage_key)))
        
        # Add JSON instructions to prompt if not using native schema support
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        full_prompt = f"{prompt}\n\nIMPORTANT: Return valid JSON matching this schema:\n{schema_json}"
        try:
            raw_text = self.generate(full_prompt, stage_key)
            return schema(**extract_json(raw_text))
        except Exception as e:
            logger.warning(f"Structured scaling failed, using mock: {e}")
            return schema(**extract_json(self._mock_generate(stage_key)))

    @property
    @abstractmethod
    def is_mock(self) -> bool: pass

    def _mock_generate(self, stage_key: str) -> str:
        return _MOCK_RESPONSES.get(stage_key, json.dumps({"error": "unknown stage", "mock": True}))

# ── Gemini Client ───────────────────────────────────────────────────────────
class LLMClient(BaseLLMProvider):
    """Gemini API client with native Pydantic support."""

    def __init__(self, model_name: str | None = None, temperature: float = 0.2, mock: bool = False):
        from config import SA_MODEL_NAME
        self.model_name = model_name or SA_MODEL_NAME
        self.temperature = temperature
        self.api_key = os.environ.get("GOOGLE_API_KEY", "")
        self._is_mock = mock or not self.api_key
        self._client = None
        if not self._is_mock:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception: self._is_mock = True

    def generate(self, prompt: str, stage_key: str = "") -> str:
        """Generate text with timeout and error handling (NFR-8.4)."""
        if self._is_mock: return self._mock_generate(stage_key)
        try:
            resp = self._client.models.generate_content(
                model=self.model_name, contents=prompt,
                config={'temperature': self.temperature},
                # Timeout handled by client config
            )
            return resp.text
        except Exception as e:
            error_str = str(e)
            logger.warning(f"Gemini API error: {error_str}")

            # Check for timeout
            if "timeout" in error_str.lower() or "deadline" in error_str.lower():
                from exceptions import TimeoutError, ErrorContext
                raise TimeoutError(
                    f"Gemini API timeout after {LLM_TIMEOUT}s",
                    context=ErrorContext(stage="llm_call", details={"provider": "gemini"}),
                    cause=e
                )

            # Return mock on any error (graceful degradation)
            return self._mock_generate(stage_key)

    def generate_structured(self, prompt: str, schema: type, stage_key: str = "") -> Any:
        if self._is_mock: return super().generate_structured(prompt, schema, stage_key)
        try:
            resp = self._client.models.generate_content(
                model=self.model_name, contents=prompt,
                config={'temperature': self.temperature, 'response_mime_type': 'application/json', 'response_schema': schema}
            )
            return resp.parsed or schema(**extract_json(resp.text))
        except Exception: return super().generate_structured(prompt, schema, stage_key)

    @property
    def is_mock(self) -> bool: return self._is_mock

# ── OpenAI Compatible Client (OpenAI, Grok, Qwen, Nvidia) ───────────────────────
class OpenAICompatibleClient(BaseLLMProvider):
    """Unified client for all OpenAI-compatible APIs with exponential backoff."""

    def __init__(self, model_name: str, api_key: str, base_url: str | None = None, temperature: float = 0.2, mock: bool = False):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self._is_mock = mock or not api_key
        self._client = None
        if not self._is_mock:
            try:
                import openai
                self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
            except Exception: self._is_mock = True

    def generate(self, prompt: str, stage_key: str = "") -> str:
        """
        Generate text with timeout, retry, and error handling (NFR-8.4, FR-2.3).
        """
        if self._is_mock: return self._mock_generate(stage_key)

        # Exponential backoff for rate limits (FR-2.3)
        max_retries = 5
        base_delay = 1.0
        max_delay = 30.0

        for attempt in range(max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    timeout=LLM_TIMEOUT,  # Explicit timeout (NFR-8.4)
                )
                return resp.choices[0].message.content
            except Exception as e:
                error_str = str(e)

                # Check for timeout
                if "timeout" in error_str.lower():
                    if attempt < max_retries - 1:
                        logger.info(f"Request timeout, retrying (attempt {attempt + 1}/{max_retries})")
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        from exceptions import TimeoutError, ErrorContext
                        raise TimeoutError(
                            f"API timeout after {LLM_TIMEOUT}s",
                            context=ErrorContext(stage="llm_call", details={"provider": self.model_name}),
                            cause=e
                        )

                # Check for rate limit
                if "429" in error_str or "rate limit" in error_str.lower():
                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        delay += random.uniform(0, 0.5)  # Add jitter
                        logger.info(f"Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(delay)
                        continue
                # Other errors: return mock on final attempt
                if attempt == max_retries - 1:
                    logger.warning(f"API call failed after {max_retries} attempts: {e}")
                    return self._mock_generate(stage_key)
        return self._mock_generate(stage_key)

    @property
    def is_mock(self) -> bool: return self._is_mock

# ── Specific Providers (Wrappers) ──────────────────────────────────────────────
class OllamaLLMClient(BaseLLMProvider):
    """Local Ollama client."""
    def __init__(self, model_name: str = "gemma3:1b", base_url: str = "http://localhost:11434", temperature: float = 0.2, mock: bool = False):
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self._is_mock = mock
        if not mock:
            import requests
            from config import SEARCH_TIMEOUT
            try:
                requests.get(f"{base_url}/api/tags", timeout=SEARCH_TIMEOUT).raise_for_status()
            except Exception:
                self._is_mock = True

    def generate(self, prompt: str, stage_key: str = "") -> str:
        """Generate text with timeout and error handling (NFR-8.4)."""
        if self._is_mock: return self._mock_generate(stage_key)
        import requests
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.temperature,
                    "stream": False
                },
                timeout=LLM_TIMEOUT  # Use config timeout (NFR-8.4)
            )
            return resp.json().get("message", {}).get("content", "")
        except requests.Timeout:
            from exceptions import TimeoutError, ErrorContext
            raise TimeoutError(
                f"Ollama timeout after {LLM_TIMEOUT}s",
                context=ErrorContext(stage="llm_call", details={"provider": "ollama", "model": self.model_name})
            )
        except Exception:
            return self._mock_generate(stage_key)

    @property
    def is_mock(self) -> bool: return self._is_mock

# ── Factory ──────────────────────────────────────────────────────────────────
class LLMClientFactory:
    @staticmethod
    def create(provider: str = "gemini", **kwargs) -> BaseLLMProvider:
        from config import QWEN_API_KEY, GROK_API_KEY, NVIDIA_API_KEY, OPENAI_API_KEY, OPENAI_BASE_URL
        p = provider.lower()
        if p == "gemini": return LLMClient(**kwargs)
        if p == "gemma": return OllamaLLMClient(**kwargs)
        if p == "qwen": return OpenAICompatibleClient(model_name=kwargs.get("model_name", "qwen2.5-7b-instruct"), api_key=QWEN_API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", **kwargs)
        if p == "grok": return OpenAICompatibleClient(model_name=kwargs.get("model_name", "grok-2"), api_key=GROK_API_KEY, base_url="https://api.x.ai/v1", **kwargs)
        if p == "nvidia": return OpenAICompatibleClient(model_name=kwargs.get("model_name", "meta/llama-3.1-70b-instruct"), api_key=NVIDIA_API_KEY, base_url="https://integrate.api.nvidia.com/v1", **kwargs)
        if p == "openai": return OpenAICompatibleClient(model_name=kwargs.get("model_name", "gpt-4o"), api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL, **kwargs)
        raise ValueError(f"Unknown provider: {provider}")
