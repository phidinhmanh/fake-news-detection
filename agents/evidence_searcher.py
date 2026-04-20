"""
evidence_searcher.py — RAG Search Orchestrator
==============================================
Orchestrates evidence retrieval from Local KB, Wikipedia, and Google (Serper).

Follows VN-4: Filter search results for Vietnamese language/relevance.
"""

from __future__ import annotations
import logging
import requests
from pydantic import BaseModel, Field

from security import contains_vietnamese

logger = logging.getLogger(__name__)

# Vietnamese language filter threshold (VN-4)
VIETNAMESE_MIN_RATIO = 0.3  # Minimum ratio of Vietnamese characters

class EvidenceItem(BaseModel):
    text: str
    source: str = "kb"
    score: float = 0.0
    stance: str = "neutral"
    is_vietnamese: bool = False  # VN-4: Track Vietnamese content

def filter_vietnamese_relevance(
    results: list[EvidenceItem],
    min_ratio: float = VIETNAMESE_MIN_RATIO,
) -> list[EvidenceItem]:
    """
    Filter search results to ensure Vietnamese relevance (VN-4).

    For Wikipedia/Wikipedia results, keeps Vietnamese language results.
    For local KB, keeps all results (assumed already Vietnamese).
    For Serper results, scores higher if containing Vietnamese.
    """
    filtered = []
    for item in results:
        # Local KB results: assume Vietnamese if not explicitly non-Vietnamese
        if item.source == "local_kb":
            filtered.append(item)
            continue

        # Check if content contains Vietnamese
        has_vietnamese = contains_vietnamese(item.text)

        if has_vietnamese:
            # High confidence for Vietnamese content
            item.is_vietnamese = True
            filtered.append(item)
        elif item.source.startswith("wiki"):
            # For Wikipedia, prefer Vietnamese Wikipedia
            if ".vi." in item.source or "vi.wikipedia" in item.source:
                item.is_vietnamese = True
                filtered.append(item)
            # Skip English Wikipedia for Vietnamese claims
        else:
            # For Serper/other, include but mark as non-Vietnamese
            item.is_vietnamese = False
            filtered.append(item)

    logger.debug(f"Filtered to {len(filtered)} Vietnamese-relevant results")
    return filtered

class EvidenceSearcher:
    """Orchestrates search across multiple providers."""

    def __init__(self, kb=None, use_wiki: bool = True, use_serper: bool = True):
        from config import SERPER_API_KEY
        self.kb = kb
        self.use_wiki = use_wiki
        self.serper_key = SERPER_API_KEY if use_serper else None

    def _wiki_search(self, query: str, limit: int = 2) -> list[dict]:
        """Search Vietnamese Wikipedia with timeout (NFR-8.4)."""
        try:
            from config import SEARCH_TIMEOUT
            url = "https://vi.wikipedia.org/w/api.php"
            params = {"action": "query", "list": "search", "srsearch": query, "srlimit": limit, "format": "json"}
            items = requests.get(url, params=params, timeout=SEARCH_TIMEOUT).json().get("query", {}).get("search", [])
            results = []
            for item in items:
                title = item["title"]
                ex = requests.get(
                    url,
                    params={"action": "query", "titles": title, "prop": "extracts", "exintro": 1, "explaintext": 1, "format": "json"},
                    timeout=SEARCH_TIMEOUT
                ).json()
                page = next(iter(ex["query"]["pages"].values()))
                if "extract" in page:
                    results.append({"text": f"Wikipedia: {title} - {page['extract'][:300]}", "source": f"wiki:{title}"})
            return results
        except Exception as e:
            logger.warning(f"Wikipedia search failed: {e}")
            return []

    def _serper_search(self, query: str, limit: int = 2) -> list[dict]:
        """Search Google via Serper with timeout (NFR-8.4)."""
        if not self.serper_key:
            return []
        try:
            from config import SEARCH_TIMEOUT
            url = "https://google.serper.dev/search"
            res = requests.post(
                url,
                json={"q": query, "num": limit},
                headers={"X-API-KEY": self.serper_key},
                timeout=SEARCH_TIMEOUT
            ).json()
            return [{"text": f"Google: {r.get('title')} - {r.get('snippet')}", "source": r.get("link")} for r in res.get("organic", [])]
        except Exception as e:
            logger.warning(f"Serper search failed: {e}")
            return []

    def search_claim(self, claim: str, top_k: int = 3) -> list[EvidenceItem]:
        """Search for evidence supporting/refuting a claim (VN-4)."""
        all_ev = []
        # 1. Local KB
        if self.kb:
            for r in self.kb.search(claim, top_k=top_k):
                all_ev.append(EvidenceItem(
                    text=r["evidence"],
                    source="local_kb",
                    score=1-r["score"],
                    is_vietnamese=True  # Local KB assumed Vietnamese
                ))

        # 2. External
        if self.use_wiki:
            for r in self._wiki_search(claim):
                all_ev.append(EvidenceItem(text=r["text"], source=r["source"], score=0.5))

        if self.serper_key:
            for r in self._serper_search(claim):
                all_ev.append(EvidenceItem(text=r["text"], source=r["source"], score=0.5))

        # 3. Filter for Vietnamese relevance (VN-4)
        all_ev = filter_vietnamese_relevance(all_ev)

        return sorted(all_ev, key=lambda x: x.score, reverse=True)[:top_k]

    def search_multi(self, claims: list[str]) -> dict[str, list[EvidenceItem]]:
        return {c: self.search_claim(c) for c in claims}
