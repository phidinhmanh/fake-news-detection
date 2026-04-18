"""
evidence_searcher.py — Agent 2: RAG Search Evidence
=====================================================
Tìm evidence cho mỗi claim: ViFactCheck KB (vector search) + Wikipedia VN + Serper API.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Search Provider Abstract Class ──────────────────────────────────────────────
class SearchProvider(ABC):
    """Abstract base class for search providers."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for evidence.

        Args:
            query: Search query.
            top_k: Number of results to return.

        Returns:
            List of dicts: {title, text, url}
        """
        pass


# ── Wikipedia VN Search ─────────────────────────────────────────────────────────
def search_wikipedia_vn(query: str, max_results: int = 3) -> list[dict]:
    """Tìm kiếm trên Wikipedia tiếng Việt (API miễn phí).

    Args:
        query: Từ khóa tìm kiếm.
        max_results: Số kết quả tối đa.

    Returns:
        List of dicts: {title, text, url}
    """
    import requests

    try:
        # Wikipedia API search
        headers = {"User-Agent": "FakeNewsDetection/1.0 (Contact: admin@example.com)"}
        search_url = "https://vi.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": max_results,
            "format": "json",
            "utf8": 1,
        }

        resp = requests.get(search_url, params=search_params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("query", {}).get("search", []):
            title = item.get("title", "")

            # Get article extract
            extract_params = {
                "action": "query",
                "titles": title,
                "prop": "extracts",
                "exintro": True,
                "explaintext": True,
                "format": "json",
                "utf8": 1,
            }

            ext_resp = requests.get(search_url, params=extract_params, headers=headers, timeout=10)
            ext_data = ext_resp.json()

            pages = ext_data.get("query", {}).get("pages", {})
            for page in pages.values():
                extract = page.get("extract", "")
                if extract:
                    results.append({
                        "title": title,
                        "text": extract[:500],
                        "url": f"https://vi.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    })

        return results

    except Exception as exc:
        logger.warning(f"⚠️ Wikipedia search failed: {exc}")
        return []


class WikipediaSearch(SearchProvider):
    """Wikipedia VN search provider."""

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search Wikipedia VN."""
        return search_wikipedia_vn(query, max_results=top_k)


class SerperSearch(SearchProvider):
    """Serper API (serpapi.com) search provider for Google search results."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Serper search provider.

        Args:
            api_key: Serper API key. If None, reads from SERPER_API_KEY env var.
        """
        import os as _os
        from config import SERPER_API_KEY

        self.api_key = api_key or SERPER_API_KEY or _os.getenv("SERPER_API_KEY", "")
        self.base_url = "https://serpapi.com/search"

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search using Serper API (Google).

        Args:
            query: Search query.
            top_k: Number of results to return.

        Returns:
            List of dicts: {title, text, url}
        """
        if not self.api_key:
            logger.warning("⚠️ Serper API key not set. Set SERPER_API_KEY env var.")
            return []

        try:
            import requests

            params = {
                "q": query,
                "api_key": self.api_key,
                "engine": "google",
                "num": top_k,
            }

            resp = requests.get(self.base_url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            results = []
            for item in data.get("organic_results", [])[:top_k]:
                results.append({
                    "title": item.get("title", ""),
                    "text": item.get("snippet", "")[:500],
                    "url": item.get("link", ""),
                })

            logger.info(f"  🌐 Serper: {len(results)} results for '{query[:50]}...'")
            return results

        except Exception as exc:
            logger.warning(f"⚠️ Serper search failed: {exc}")
            return []


# ── Schema ─────────────────────────────────────────────────────────
class EvidenceItem(BaseModel):
    """Một mảnh evidence tìm được."""

    text: str = Field(description="Nội dung evidence")
    source: str = Field(default="knowledge_base", description="Nguồn: knowledge_base / wikipedia / serper")
    relevance_score: float = Field(default=0.0, description="Điểm liên quan")
    stance: str = Field(default="neutral", description="support / refute / neutral")


class ClaimEvidence(BaseModel):
    """Evidence cho 1 claim."""

    claim_text: str = Field(description="Claim gốc")
    evidences: list[EvidenceItem] = Field(default_factory=list)
    has_evidence: bool = Field(default=False)


class EvidenceSearchResult(BaseModel):
    """Tổng hợp evidence cho tất cả claims."""

    claim_evidences: list[ClaimEvidence] = Field(default_factory=list)
    total_evidence_found: int = Field(default=0)
    sources_used: list[str] = Field(default_factory=list)


# ── Agent ──────────────────────────────────────────────────────────
class EvidenceSearcher:
    """Agent 2: Tìm evidence cho claims.

    Workflow:
        1. Search ViFactCheck knowledge base (vector similarity)
        2. Search Wikipedia VN / Serper (keyword search)
        3. Combine & rank results

    Usage:
        from agents.knowledge_base import KnowledgeBase
        from agents.evidence_searcher import EvidenceSearcher

        kb = KnowledgeBase()
        searcher = EvidenceSearcher(kb)
        result = searcher.search_claims(["Claim 1", "Claim 2"])
    """

    def __init__(
        self,
        knowledge_base=None,
        use_wikipedia: bool = True,
        use_serper: bool = True,
        top_k: int = 5,
    ):
        """
        Args:
            knowledge_base: KnowledgeBase instance (optional).
            use_wikipedia: Có search Wikipedia VN hay không.
            use_serper: Có search Serper API hay không.
            top_k: Số evidence tối đa per claim.
        """
        self.kb = knowledge_base
        self.top_k = top_k

        # Initialize search providers
        self._providers: list[SearchProvider] = []
        if use_wikipedia:
            self._providers.append(WikipediaSearch())
        if use_serper:
            serper = SerperSearch()
            # Only add if API key is available
            if serper.api_key:
                self._providers.append(serper)
            else:
                logger.info("⚠️ Serper API key not set, skipping Serper search.")

    def search_single_claim(self, claim_text: str) -> ClaimEvidence:
        """Tìm evidence cho 1 claim.

        Args:
            claim_text: Nội dung claim.

        Returns:
            ClaimEvidence object.
        """
        evidences: list[EvidenceItem] = []

        # 1. Search Knowledge Base
        if self.kb:
            try:
                kb_results = self.kb.search(claim_text, top_k=self.top_k)
                for r in kb_results:
                    evidences.append(EvidenceItem(
                        text=r.get("evidence", r.get("claim", "")),
                        source="vifactcheck_kb",
                        relevance_score=1.0 - min(r.get("score", 1.0), 1.0),  # Convert distance to score
                        stance=self._infer_stance(r.get("label", "neutral")),
                    ))
                logger.info(f"  📚 KB: {len(kb_results)} results for '{claim_text[:50]}...'")
            except Exception as exc:
                logger.warning(f"⚠️ KB search failed: {exc}")

        # 2. Search external providers (Wikipedia, Serper)
        for provider in self._providers:
            provider_name = provider.__class__.__name__
            try:
                results = provider.search(claim_text, top_k=3)
                for r in results:
                    evidences.append(EvidenceItem(
                        text=f"[{provider_name}] {r['title']}: {r['text']}",
                        source=r.get("url", provider_name.lower()),
                        relevance_score=0.5,  # Default relevance
                        stance="neutral",
                    ))
                logger.info(f"  📖 {provider_name}: {len(results)} results")
            except Exception as exc:
                logger.warning(f"⚠️ {provider_name} search failed: {exc}")

        # Sort by relevance
        evidences.sort(key=lambda e: e.relevance_score, reverse=True)
        evidences = evidences[:self.top_k]

        return ClaimEvidence(
            claim_text=claim_text,
            evidences=evidences,
            has_evidence=len(evidences) > 0,
        )

    def search_claims(self, claims: list[str]) -> EvidenceSearchResult:
        """Tìm evidence cho nhiều claims.

        Args:
            claims: List of claim strings.

        Returns:
            EvidenceSearchResult object.
        """
        logger.info(f"🔎 [Agent 2] Searching evidence for {len(claims)} claims...")

        claim_evidences: list[ClaimEvidence] = []
        sources_used: set[str] = set()
        total = 0

        for claim_text in claims:
            ce = self.search_single_claim(claim_text)
            claim_evidences.append(ce)
            total += len(ce.evidences)
            for e in ce.evidences:
                sources_used.add(e.source)

        logger.info(f"✅ [Agent 2] Found {total} total evidence pieces from {len(sources_used)} sources")

        return EvidenceSearchResult(
            claim_evidences=claim_evidences,
            total_evidence_found=total,
            sources_used=list(sources_used),
        )

    @staticmethod
    def _infer_stance(label: str) -> str:
        """Map ViFactCheck label → stance."""
        mapping = {
            "real": "support",
            "fake": "refute",
            "suspicious": "neutral",
            "supported": "support",
            "refuted": "refute",
        }
        return mapping.get(label.lower(), "neutral")
