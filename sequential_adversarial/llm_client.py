"""
llm_client.py — Thin wrapper around Google Gemini API
======================================================
Handles API key from env, falls back to mock mode when key is absent (for CI/tests).
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Optional


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
    })
}


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

    @property
    def is_mock(self) -> bool:
        return self.mock


def extract_json(text: str) -> dict:
    """
    Safely extract a JSON object from an LLM response string.
    Handles cases where the model wraps JSON in markdown code blocks.
    """
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences if present
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Last resort: find first { ... } block
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {"error": "Failed to parse LLM response", "raw": text[:500]}
