"""
llm_client.py — Thin wrapper around Google Gemini API
======================================================
Handles API key from env, falls back to mock mode when key is absent (for CI/tests).
Supports multiple LLM providers: Gemini, Gemma (Ollama), Qwen, Grok via factory pattern.
"""

from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Optional, Any


# ── Mock responses for each stage (used when GOOGLE_API_KEY is not set) ────────
_MOCK_RESPONSES: dict[str, str] = {
    "lead_investigator": json.dumps({
        "claims": [
            {
                "text": "The stock market will crash by 50% next week",
                "suspicion_score": 0.9,
                "loaded_language": ["crash", "50%"]
            },
            {
                "text": "Government officials are hiding the truth from the public",
                "suspicion_score": 0.85,
                "loaded_language": ["hiding the truth"]
            }
        ],
        "overall_manipulation_score": 0.87,
        "summary": "[MOCK] High suspicion content detected. Multiple loaded language patterns found."
    }),
    "data_analyst": json.dumps({
        "claim_analyses": [
            {
                "claim": {
                    "text": "The stock market will crash by 50% next week",
                    "suspicion_score": 0.9,
                    "loaded_language": ["crash", "50%"]
                },
                "sources": [
                    {
                        "url": "https://reuters.com/markets/mock",
                        "stance": "refute",
                        "reliability": 0.9,
                        "excerpt": "[MOCK] No credible financial analysts predict such a dramatic drop."
                    }
                ],
                "verdict": "refuted"
            }
        ],
        "summary": "[MOCK] Claims checked against 3 sources. Main claim is refuted by credible sources."
    }),
    "bias_auditor": json.dumps({
        "framing": "Economic fear and political conspiracy",
        "distortion_detected": True,
        "distortion_type": "false urgency",
        "adversarial_notes": "[MOCK] The analyst's conclusion appears correct, but source diversity is limited. The narrative leverages fear-based framing to bypass rational evaluation.",
        "analyst_blind_spots": ["Missing local context", "No expert quotes verified"]
    }),
    "synthesizer": json.dumps({
        "conclusion": "False",
        "confidence": 0.85,
        "evidence_summary": "[MOCK] The primary claim is refuted by 3 independent credible sources.",
        "bias_summary": "[MOCK] Content uses fear-based framing and loaded language to appear credible.",
        "key_findings": [
            "Primary claim directly refuted by Reuters",
            "High loaded language density (2 phrases)",
            "False urgency detected in narrative framing"
        ],
        "markdown_report": "## Verity Report\n**Conclusion:** FALSE ❌\n\n**Evidence:** Mock analysis complete.\n\n**Bias:** Fear-based framing detected."
    }),
    # ── New Agent Pipeline Mocks ──────────────────────────────────────
    "claim_extractor": json.dumps({
        "claims": [
            {
                "text": "Vaccine COVID-19 gây ra hàng nghìn ca tử vong",
                "importance": 0.95,
                "category": "health"
            },
            {
                "text": "Các hãng dược phẩm che giấu tác dụng phụ",
                "importance": 0.85,
                "category": "health"
            }
        ],
        "article_summary": "[MOCK] Bài viết chứa thông tin chưa được kiểm chứng về vaccine COVID-19.",
        "overall_credibility_hint": 0.2
    }),
    "reasoning_scorer": json.dumps({
        "fake_score": 82,
        "label": "Fake",
        "confidence": 0.85,
        "reasoning_steps": [
            "[MOCK] Bước 1: Claim chính về tử vong do vaccine thiếu nguồn đáng tin cậy",
            "[MOCK] Bước 2: Không tìm thấy nghiên cứu bị rò rỉ nào từ nguồn uy tín",
            "[MOCK] Bước 3: WHO và CDC đã bác bỏ thông tin này nhiều lần"
        ],
        "explanation": "[MOCK] Bài viết chứa nhiều yếu tố đáng ngờ: sử dụng ngôn ngữ gây sốc (KHẨN CẤP, hàng nghìn ca tử vong), thiếu trích dẫn nguồn cụ thể, và kêu gọi chia sẻ ngay. Các tổ chức y tế quốc tế đã bác bỏ thông tin tương tự.",
        "evidence_citations": [
            {
                "claim": "Vaccine gây tử vong",
                "verdict": "REFUTED",
                "supporting_evidence": "[MOCK] WHO: Vaccine COVID-19 an toàn và hiệu quả"
            }
        ],
        "risk_factors": ["Ngôn ngữ gây sốc", "Thiếu nguồn đáng tin cậy", "Kêu gọi chia sẻ gấp"]
    })
}


# ── Base LLM Provider Abstract Class ───────────────────────────────────────────
class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, stage_key: str = "") -> str:
        """Generate a response for the given prompt."""
        pass

    @abstractmethod
    def generate_structured(self, prompt: str, schema: type, stage_key: str = "") -> Any:
        """Generate a structured response conforming to a Pydantic schema."""
        pass

    @property
    @abstractmethod
    def is_mock(self) -> bool:
        """Return True if using mock mode."""
        pass


class LLMClient:
    """
    Gemini API client with automatic mock fallback.

    Usage:
        client = LLMClient()           # uses GOOGLE_API_KEY env var
        client = LLMClient(mock=True)  # force mock mode (for testing)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.2,
        mock: bool = False,
    ):
        from config import SA_MODEL_NAME  # lazy import to avoid circular deps
        self.model_name = model_name or SA_MODEL_NAME
        self.temperature = temperature
        self.api_key = os.environ.get("GOOGLE_API_KEY", "")
        self.mock = mock or not self.api_key

        self._model = None
        if not self.mock:
            self._init_gemini()

    def _init_gemini(self) -> None:
        """Initialize the Gemini generative model."""
        try:
            import google.generativeai as genai  # type: ignore

            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": self.temperature,
                    "response_mime_type": "application/json",
                },
            )
        except ImportError:
            print(
                "[LLMClient] WARNING: google-generativeai not installed. "
                "Falling back to mock mode. Run: pip install google-generativeai"
            )
            self.mock = True

    def generate(self, prompt: str, stage_key: str = "") -> str:
        """
        Generate a response for the given prompt.

        Args:
            prompt: The full prompt to send to the LLM.
            stage_key: Stage identifier used for mock responses
                       (e.g. 'lead_investigator', 'data_analyst').

        Returns:
            Raw response string (should be valid JSON per the system prompt).
        """
        if self.mock:
            return self._mock_generate(stage_key)

        try:
            response = self._model.generate_content(prompt)
            return response.text
        except Exception as exc:
            print(f"[LLMClient] API error: {exc}. Falling back to mock.")
            return self._mock_generate(stage_key)

    def _mock_generate(self, stage_key: str) -> str:
        """Return a canned mock response for the given stage."""
        return _MOCK_RESPONSES.get(stage_key, json.dumps({"error": "unknown stage", "mock": True}))

    def generate_structured(self, prompt: str, schema: type, stage_key: str = "") -> any:
        """
        Calls the LLM and strictly parses the output into a predefined Pydantic model (`schema`).
        Handles failures gracefully by returning raw JSON data that pipeline.py will catch,
        or throwing explicit errors for bad parsing.

        Args:
            prompt: Original prompt instructing JSON schema.
            schema: Pydantic BaseModel class (e.g. InvestigationResult).
            stage_key: Used for mock responses.

        Returns:
            An instantiated Pydantic model object, strictly conforming to the schema.
        """
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        full_prompt = f"{prompt}\n\nIMPORTANT: You must return valid JSON that strictly adheres to the following schema:\n{schema_json}"

        raw_text = self.generate(full_prompt, stage_key)
        data_dict = extract_json(raw_text)

        # Biến dictionary thành Pydantic Model (Tự động validate cấu trúc)
        # Nếu LLM trả về JSON bị thiếu / móp méo trường, schema(**data_dict) sẽ tốn throw ValueError.
        return schema(**data_dict)

    @property
    def is_mock(self) -> bool:
        return self.mock


# ── Ollama LLM Client (Gemma, etc.) ──────────────────────────────────────────────
class OllamaLLMClient(BaseLLMProvider):
    """LLM client using Ollama for local models (Gemma, etc.)."""

    def __init__(
        self,
        model_name: str = "gemma3:1b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
        mock: bool = False,
    ):
        from config import OLLAMA_BASE_URL
        self.model_name = model_name
        self.base_url = base_url or OLLAMA_BASE_URL
        self.temperature = temperature
        self.mock = mock or not self._check_ollama()

        self._client = None
        if not self.mock:
            self._init_ollama()

    def _check_ollama(self) -> bool:
        """Check if Ollama is running."""
        try:
            import requests
            resp = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def _init_ollama(self) -> None:
        """Initialize Ollama client."""
        try:
            from ollama import ChatResponse, Message
            self._client = type('OllamaClient', (), {})()
            self._client._base_url = self.base_url
        except ImportError:
            print("[OllamaLLMClient] WARNING: ollama package not installed. Run: pip install ollama")
            self.mock = True

    def generate(self, prompt: str, stage_key: str = "") -> str:
        """Generate response via Ollama API."""
        if self.mock:
            return self._mock_generate(stage_key)

        try:
            import requests

            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "stream": False,
            }
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")
        except Exception as exc:
            print(f"[OllamaLLMClient] API error: {exc}. Falling back to mock.")
            return self._mock_generate(stage_key)

    def _mock_generate(self, stage_key: str) -> str:
        """Return mock response."""
        return _MOCK_RESPONSES.get(stage_key, json.dumps({"error": "unknown stage", "mock": True}))

    def generate_structured(self, prompt: str, schema: type, stage_key: str = "") -> any:
        """Generate structured response via Ollama."""
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        full_prompt = f"{prompt}\n\nIMPORTANT: Return valid JSON matching this schema:\n{schema_json}"

        raw_text = self.generate(full_prompt, stage_key)
        data_dict = extract_json(raw_text)
        return schema(**data_dict)

    @property
    def is_mock(self) -> bool:
        return self.mock


# ── Qwen LLM Client ──────────────────────────────────────────────────────────────
class QwenLLMClient(BaseLLMProvider):
    """LLM client using Qwen via DashScope or OpenAI-compatible API."""

    def __init__(
        self,
        model_name: str = "qwen2.5-7b-instruct",
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature: float = 0.2,
        mock: bool = False,
    ):
        from config import QWEN_API_KEY
        self.model_name = model_name
        self.api_key = api_key or QWEN_API_KEY
        self.base_url = base_url
        self.temperature = temperature
        self.mock = mock or not self.api_key

        self._client = None
        if not self.mock:
            self._init_qwen()

    def _init_qwen(self) -> None:
        """Initialize Qwen client."""
        try:
            import openai
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        except ImportError:
            print("[QwenLLMClient] WARNING: openai package not installed.")
            self.mock = True

    def generate(self, prompt: str, stage_key: str = "") -> str:
        """Generate response via Qwen API."""
        if self.mock:
            return self._mock_generate(stage_key)

        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            return response.choices[0].message.content
        except Exception as exc:
            print(f"[QwenLLMClient] API error: {exc}. Falling back to mock.")
            return self._mock_generate(stage_key)

    def _mock_generate(self, stage_key: str) -> str:
        """Return mock response."""
        return _MOCK_RESPONSES.get(stage_key, json.dumps({"error": "unknown stage", "mock": True}))

    def generate_structured(self, prompt: str, schema: type, stage_key: str = "") -> any:
        """Generate structured response via Qwen."""
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        full_prompt = f"{prompt}\n\nIMPORTANT: Return valid JSON matching this schema:\n{schema_json}"

        raw_text = self.generate(full_prompt, stage_key)
        data_dict = extract_json(raw_text)
        return schema(**data_dict)

    @property
    def is_mock(self) -> bool:
        return self.mock


# ── Grok LLM Client ─────────────────────────────────────────────────────────────
class GrokLLMClient(BaseLLMProvider):
    """LLM client using Grok via xAI API."""

    def __init__(
        self,
        model_name: str = "grok-2",
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        mock: bool = False,
    ):
        from config import GROK_API_KEY
        self.model_name = model_name
        self.api_key = api_key or GROK_API_KEY
        self.temperature = temperature
        self.mock = mock or not self.api_key

        self._client = None
        if not self.mock:
            self._init_grok()

    def _init_grok(self) -> None:
        """Initialize Grok client."""
        try:
            import openai
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://api.x.ai/v1",
            )
        except ImportError:
            print("[GrokLLMClient] WARNING: openai package not installed.")
            self.mock = True

    def generate(self, prompt: str, stage_key: str = "") -> str:
        """Generate response via Grok API."""
        if self.mock:
            return self._mock_generate(stage_key)

        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            return response.choices[0].message.content
        except Exception as exc:
            print(f"[GrokLLMClient] API error: {exc}. Falling back to mock.")
            return self._mock_generate(stage_key)

    def _mock_generate(self, stage_key: str) -> str:
        """Return mock response."""
        return _MOCK_RESPONSES.get(stage_key, json.dumps({"error": "unknown stage", "mock": True}))

    def generate_structured(self, prompt: str, schema: type, stage_key: str = "") -> any:
        """Generate structured response via Grok."""
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        full_prompt = f"{prompt}\n\nIMPORTANT: Return valid JSON matching this schema:\n{schema_json}"

        raw_text = self.generate(full_prompt, stage_key)
        data_dict = extract_json(raw_text)
        return schema(**data_dict)

    @property
    def is_mock(self) -> bool:
        return self.mock


# ── Nvidia LLM Client ──────────────────────────────────────────────────────────
class NvidiaLLMClient(BaseLLMProvider):
    """LLM client using Nvidia NIM via langchain_nvidia_ai_endpoints."""

    def __init__(
        self,
        model_name: str = "qwen/qwen3.5-122b-a10b",
        api_key: Optional[str] = None,
        temperature: float = 0.6,
        mock: bool = False,
    ):
        from config import NVIDIA_API_KEY
        self.model_name = model_name
        # Hard code API if user explicitly requests it otherwise use environment config
        self.api_key = api_key or NVIDIA_API_KEY
        self.temperature = temperature
        self.mock = mock or not self.api_key

        self._client = None
        if not self.mock:
            self._init_nvidia()

    def _init_nvidia(self) -> None:
        """Initialize Nvidia client."""
        try:
            from langchain_nvidia_ai_endpoints import ChatNVIDIA
            self._client = ChatNVIDIA(
                model=self.model_name,
                api_key=self.api_key,
                temperature=self.temperature,
                top_p=0.95,
                max_completion_tokens=16384,
            )
        except ImportError:
            print("[NvidiaLLMClient] WARNING: langchain_nvidia_ai_endpoints not installed.")
            self.mock = True

    def generate(self, prompt: str, stage_key: str = "") -> str:
        """Generate response via Nvidia NIM API."""
        if self.mock:
            return self._mock_generate(stage_key)

        try:
            lc_messages = [{"role": "user", "content": prompt}]
            full_content = ""
            for chunk in self._client.stream(lc_messages, chat_template_kwargs={"enable_thinking": True}):
                # Currently we aren't storing the reasoning_content out of the pipeline, 
                # but we can print it for debug visibility as the user requested in the code snippet.
                if chunk.additional_kwargs and "reasoning_content" in chunk.additional_kwargs:
                    print(chunk.additional_kwargs["reasoning_content"], end="")
                if chunk.content:
                    full_content += chunk.content
            return full_content
        except Exception as exc:
            print(f"[NvidiaLLMClient] API error: {exc}. Falling back to mock.")
            return self._mock_generate(stage_key)

    def _mock_generate(self, stage_key: str) -> str:
        return _MOCK_RESPONSES.get(stage_key, json.dumps({"error": "unknown stage", "mock": True}))

    def generate_structured(self, prompt: str, schema: type, stage_key: str = "") -> any:
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        full_prompt = f"{prompt}\n\nIMPORTANT: Return valid JSON matching this schema:\n{schema_json}"
        raw_text = self.generate(full_prompt, stage_key)
        data_dict = extract_json(raw_text)
        return schema(**data_dict)

    @property
    def is_mock(self) -> bool:
        return self.mock


# ── OpenAI LLM Client ──────────────────────────────────────────────────────────
class OpenAILLMClient(BaseLLMProvider):
    """LLM client using OpenAI compatible API for gpt-oss-20b and gpt-oss-120b."""

    def __init__(
        self,
        model_name: str = "gpt-oss-120b",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
        mock: bool = False,
    ):
        from config import OPENAI_API_KEY, OPENAI_BASE_URL
        self.model_name = model_name
        self.api_key = api_key or OPENAI_API_KEY
        self.base_url = base_url or OPENAI_BASE_URL
        self.temperature = temperature
        self.mock = mock or not self.api_key

        self._client = None
        if not self.mock:
            self._init_openai()

    def _init_openai(self) -> None:
        """Initialize OpenAI client."""
        try:
            import openai
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url if self.base_url else None,
            )
        except ImportError:
            print("[OpenAILLMClient] WARNING: openai package not installed.")
            self.mock = True

    def generate(self, prompt: str, stage_key: str = "") -> str:
        """Generate response via OpenAI API."""
        if self.mock:
            return self._mock_generate(stage_key)

        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            return response.choices[0].message.content
        except Exception as exc:
            print(f"[OpenAILLMClient] API error: {exc}. Falling back to mock.")
            return self._mock_generate(stage_key)

    def _mock_generate(self, stage_key: str) -> str:
        return _MOCK_RESPONSES.get(stage_key, json.dumps({"error": "unknown stage", "mock": True}))

    def generate_structured(self, prompt: str, schema: type, stage_key: str = "") -> any:
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        full_prompt = f"{prompt}\n\nIMPORTANT: Return valid JSON matching this schema:\n{schema_json}"

        raw_text = self.generate(full_prompt, stage_key)
        data_dict = extract_json(raw_text)
        return schema(**data_dict)

    @property
    def is_mock(self) -> bool:
        return self.mock


# ── LLM Client Factory ──────────────────────────────────────────────────────────
class LLMClientFactory:
    """Factory for creating LLM client instances with multi-provider support.

    Usage:
        # Default (Gemini)
        client = LLMClientFactory.create()

        # Explicit provider
        client = LLMClientFactory.create("gemini")
        client = LLMClientFactory.create("gemma")
        client = LLMClientFactory.create("qwen")
        client = LLMClientFactory.create("grok")
    """

    PROVIDERS = {
        "gemini": LLMClient,
        "gemma": OllamaLLMClient,
        "qwen": QwenLLMClient,
        "grok": GrokLLMClient,
        "nvidia": NvidiaLLMClient,
        "openai": OpenAILLMClient,
    }

    @staticmethod
    def create(provider: str = "gemini", **kwargs) -> BaseLLMProvider:
        """Create an LLM client for the specified provider.

        Args:
            provider: One of 'gemini', 'gemma', 'qwen', 'grok'.
            **kwargs: Additional arguments passed to the provider constructor.

        Returns:
            BaseLLMProvider instance.

        Raises:
            ValueError: If provider is not supported.
        """
        provider = provider.lower()
        if provider not in LLMClientFactory.PROVIDERS:
            available = ", ".join(LLMClientFactory.PROVIDERS.keys())
            raise ValueError(
                f"Unknown provider '{provider}'. Available: {available}"
            )

        client_class = LLMClientFactory.PROVIDERS[provider]
        return client_class(**kwargs)

    @classmethod
    def list_providers(cls) -> list[str]:
        """Return list of available provider names."""
        return list(cls.PROVIDERS.keys())


def extract_json(text: str) -> dict:
    """Extract JSON from raw LLM output, handling markdown code blocks."""
    text = text.strip()
    
    # Remove markdown code block syntax if present
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
        
    text = text.strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object using regex as a fallback
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not parse valid JSON from LLM output: {text}")
